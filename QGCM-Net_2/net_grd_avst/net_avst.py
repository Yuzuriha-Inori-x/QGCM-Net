import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


class CMA(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim=512, output_dim=512, reduction_factor=16, num_conv_group=4, num_tokens=64):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))
        self.down_sample_size = input_dim // reduction_factor
        self.my_tokens = nn.Parameter(torch.zeros((num_tokens, input_dim)))
        self.gate_av = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU(inplace=True)
        self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=num_conv_group,
                                      bias=False)
        self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=num_conv_group,
                                    bias=False)
        self.bn1 = nn.BatchNorm2d(self.down_sample_size)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.ln_before = nn.LayerNorm(output_dim)
        self.ln_post = nn.LayerNorm(output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        """

        Args:
            x: [bs, 512, 36, 1]
            y: [bs, 512, 36, 1]

        Returns:

        """
        rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))  # [bs, 64, 512]
        att_y2tk = torch.bmm(rep_token, y.squeeze(-1))  # [bs, 64, 36]
        att_y2tk = F.softmax(att_y2tk, dim=-1)  # [bs, 64, 36]
        rep_token_res = torch.bmm(att_y2tk, y.squeeze(-1).permute(0, 2, 1))  # [bs, 64, 512]
        rep_token = rep_token + rep_token_res  # [bs, 64, 512]
        att_tk2x = torch.bmm(x.squeeze(-1).permute(0, 2, 1), rep_token.permute(0, 2, 1))  # [bs, 36, 64]
        att_tk2x = F.softmax(att_tk2x, dim=-1)  # [bs, 36, 64]
        x_res = torch.bmm(att_tk2x, rep_token).permute(0, 2, 1).unsqueeze(-1)  # [bs, 36, 512]
        x = x + self.gate_av * x_res.contiguous()  # [bs, 36, 512, 1]
        x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)  # [bs, 512, 36, 1]
        z = self.down_sampler(x)  # [bs, 32, 36, 1]
        z = self.bn1(z)  # [bs, 32, 36, 1]
        z = self.activation(z)  # [bs, 32, 36, 1]
        output = self.up_sampler(z)  # [bs, 512, 36, 1]
        output = self.bn2(output)  # [bs, 512, 36, 1]
        output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)
        output = self.gate * output
        return output  # [bs, 512, 36, 1]


class AVQA_Fusion_Net(nn.Module):
    def __init__(self):
        super(AVQA_Fusion_Net, self).__init__()
        # for visual_posi:
        self.fc_v1_posi = nn.Linear(1536, 512)

        # for visual_nega:
        self.fc_v1_nega = nn.Linear(1536, 512)

        # for audio:
        self.fc_a1 = nn.Linear(1536, 512)

        # for question:
        self.fc_q1 = nn.Linear(512, 512)

        # CMA
        self.CMA_visual_posi = CMA()
        self.CMA_visual_nega = CMA()
        self.CMA_audio = CMA()

        # for av
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

        # for query on visual
        self.attn_v = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear11 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(512)

        # for query on audio
        self.attn_a = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear21 = nn.Linear(512, 512)
        self.linear22 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(512)

        # for query on audio-visual
        self.fc_av = nn.Linear(1024, 512)
        self.attn_av = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear31 = nn.Linear(512, 512)
        self.dropout5 = nn.Dropout(0.1)
        self.linear32 = nn.Linear(512, 512)
        self.dropout6 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(512)

        # a v fusion
        self.tanh = nn.Tanh()
        self.fc_fusion = nn.Linear(1024, 512)
        self.fc_fusion_av = nn.Linear(512, 512)
        self.fc_ans1 = nn.Linear(1024, 512)
        self.fc_ans2 = nn.Linear(512, 42)

    def forward(self, audio, visual_posi, visual_nega, question):
        """
        Args:
            audio: [bs, t, 36, 1536]
            visual_posi: [bs, t, 36, 1536]
            visual_nega: [bs t, 36, 1536]
            question: [B, 77, 512]
        Returns:
        """
        question_pure = question  # [bs, 77, 512]
        bs, t, n, c = visual_posi.size()  # [bs, t, 36, 1536]

        visual_posi = rearrange(visual_posi, 'b t n c -> (b t) n c')  # [bs*t, 36, 1536]
        visual_posi = self.fc_v1_posi(visual_posi)  # [bs*t, 36, 512]

        visual_nega = rearrange(visual_nega, 'b t n c -> (b t) n c')  # [bs*t, h*w, c]
        visual_nega = self.fc_v1_nega(visual_nega)  # [bs*t, 36, 512]

        audio = rearrange(audio, 'b t n c -> (b t) n c')  # [bs*t, h*w, c]
        audio = self.fc_a1(audio)  # [bs*t, 36, 512]
        audio_pure = audio

        question = repeat(question, 'b len dim -> b t len dim', t=t)  # [bs, t, 77,512]
        question = repeat(question, 'b t len dim -> (b t) len dim')  # [bs*t, 77,512]
        question = self.fc_q1(question)  # [bs*t, 77,512]

        ###########################################################
        # question 2 visual posi CMA
        visual_posi_res1 = self.CMA_visual_posi(visual_posi.permute(0, 2, 1).unsqueeze(-1),
                                                question.permute(0, 2, 1).unsqueeze(-1))  # [bs*t, 512, 36, 1]
        visual_posi_res1 = visual_posi + visual_posi_res1.squeeze(-1).permute(0, 2, 1)  # [bs*t, 36, 512]

        # question 2 visual nega CMA
        visual_nega_res1 = self.CMA_visual_nega(visual_nega.permute(0, 2, 1).unsqueeze(-1),
                                                question.permute(0, 2, 1).unsqueeze(-1))  # [bs*t, 512, 36, 1]
        visual_nega_res1 = visual_nega + visual_nega_res1.squeeze(-1).permute(0, 2, 1)  # [bs*t, 36, 512]

        # question 2 audio CMA
        audio_res1 = self.CMA_audio(audio.permute(0, 2, 1).unsqueeze(-1),
                                    question.permute(0, 2, 1).unsqueeze(-1))  # [bs*t, 512, 36, 1]
        audio_res1 = audio + audio_res1.squeeze(-1).permute(0, 2, 1)  # [bs*t, 36, 512]

        ###########################################################
        # audio 2 visual posi CMA
        visual_posi_res2 = self.CMA_visual_posi(visual_posi_res1.permute(0, 2, 1).unsqueeze(-1),
                                                audio_res1.permute(0, 2, 1).unsqueeze(-1))  # [bs*t, 512, 36, 1]
        visual_posi_res2 = visual_posi_res1 + visual_posi_res2.squeeze(-1).permute(0, 2, 1)  # [bs*t, 36, 512]

        # audio 2 visual nega CMA
        visual_nega_res2 = self.CMA_visual_nega(visual_nega_res1.permute(0, 2, 1).unsqueeze(-1),
                                                audio_res1.permute(0, 2, 1).unsqueeze(-1))  # [bs*t, 512, 36, 1]
        visual_nega_res2 = visual_nega_res1 + visual_nega_res2.squeeze(-1).permute(0, 2, 1)  # [bs*t, 36, 512]

        # visual 2 audio
        audio_res2 = self.CMA_audio(audio_res1.permute(0, 2, 1).unsqueeze(-1),
                                    visual_posi_res2.permute(0, 2, 1).unsqueeze(-1))  # [bs*t, 512, 36, 1]
        audio_res2 = audio_res1 + audio_res2.squeeze(-1).permute(0, 2, 1)  # [bs*t, 36, 512]

        ###########################################################
        # audi-visual posi
        feat = torch.cat((audio_pure, visual_posi_res2), dim=-1)  # [bs*t, 36, 512*2]
        feat = torch.mean(feat, dim=1)  # [bs*t, 512*2]

        feat = F.relu(self.fc1(feat))  # (1024, 512)
        feat = F.relu(self.fc2(feat))  # (512, 256)
        feat = F.relu(self.fc3(feat))  # (256, 128)
        out_match_posi = self.fc4(feat)  # (128, 2)

        ###########################################################
        # audi-visual nega
        feat = torch.cat((audio_pure, visual_nega_res2), dim=-1)  # [bs*t, 36, 512*2]
        feat = torch.mean(feat, dim=1)  # [bs*t, 512*2]

        feat = F.relu(self.fc1(feat))  # (1024, 512)
        feat = F.relu(self.fc2(feat))  # (512, 256)
        feat = F.relu(self.fc3(feat))  # (256, 128)
        out_match_nega = self.fc4(feat)  # (128, 2)

        ###########################################################
        question_avg = torch.mean(question_pure, dim=1)  # [bs, 512]
        # print('question_av:', question_avg.shape)
        question_avg_att = question_avg.unsqueeze(0)  # [1, bs, 512]
        # print('question_avg_att:', question_avg_att.shape)

        visual_feat = torch.mean(visual_posi_res2, dim=1)  # [bs*t, 512]
        visual_feat_be = rearrange(visual_feat, '(b t) c -> b t c', t=t)  # [bs, t, 512]
        visual_feat = visual_feat_be.permute(1, 0, 2)  # [t, bs, 512]
        # print('visual_feat:', visual_feat.shape)

        audio_feat = torch.mean(audio_res2, dim=1)  # [bs*t, 512]
        audio_feat_be = rearrange(audio_feat, '(b t) c -> b t c', t=t)  # [bs, t, 512]
        audio_feat = audio_feat_be.permute(1, 0, 2)  # [t, bs ,512]
        # print('audio_feat:', audio_feat.shape)

        # attention, question as qurey on visual_feat
        visual_feat_att = \
            self.attn_v(question_avg_att, visual_feat, visual_feat, attn_mask=None, key_padding_mask=None)[
                0].squeeze(0)  # [bs, 512]

        # print('visual_feat_att:', visual_feat_att.shape)
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))  # [bs, 512]
        visual_feat_att = visual_feat_att + self.dropout2(src)  # [bs, 512]
        visual_feat_att = self.norm1(visual_feat_att)  # [bs, 512]

        # attention, question as qurey on audio_feat
        audio_feat_att = \
            self.attn_a(question_avg_att, audio_feat, audio_feat, attn_mask=None, key_padding_mask=None)[
                0].squeeze(0)  # [bs, 512]
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))  # [bs, 512]
        audio_feat_att = audio_feat_att + self.dropout4(src)  # [bs, 512]
        audio_feat_att = self.norm2(audio_feat_att)  # [bs, 512]

        # attention, question as qurey on audio-visual feat
        visual_audio = torch.cat((visual_feat_be, audio_feat_be), dim=-1)  # [bs, t, 1024]
        visual_audio_be = self.fc_av(visual_audio)  # [bs, t, 512]
        visual_audio_feat = visual_audio_be.permute(1, 0, 2)  # [t, bs, 512]
        visual_audio_att = \
            self.attn_av(question_avg_att, visual_audio_feat, visual_audio_feat, attn_mask=None, key_padding_mask=None)[
                0].squeeze(0)  # [bs, 512]
        src = self.linear32(self.dropout5(F.relu(self.linear31(visual_audio_att))))  # [bs, 512]
        visual_audio_att = visual_audio_att + self.dropout6(src)  # [bs, 512]
        visual_audio_att = self.norm3(visual_audio_att)  # [bs, 512]

        new_audio = audio_feat_att + audio_feat_be.mean(dim=-2).squeeze()  # [bs, 512]
        new_visual = visual_feat_att + visual_feat_be.mean(dim=-2).squeeze()  # [bs, 512]
        new_visual_audio = visual_audio_att + visual_audio_be.mean(dim=-2).squeeze()  # [bs, 512]

        # audio visual fusion
        feat1 = torch.cat([new_audio, new_visual], dim=-1)  # [bs, 1024]
        feat1 = self.fc_fusion(self.tanh(feat1))  # [bs, 512]
        feat2 = self.fc_fusion_av(self.tanh(new_visual_audio))  # [bs, 512]

        # fusion with question
        combined_feature1 = torch.mul(feat1, question_avg)  # [bs, 512]
        combined_feature2 = torch.mul(feat2, question_avg)  # [bs, 512]
        combined_feature = torch.cat((combined_feature1, combined_feature2), dim=-1)  # [bs, 1024]

        combined_feature = self.tanh(combined_feature)  # [bs, 1024]
        combined_feature = self.fc_ans1(combined_feature)  # [bs, 512]

        combined_feature = self.tanh(combined_feature)  # [bs, 512]
        out_qa = self.fc_ans2(combined_feature)  # [batch_size, ans_vocab_size] = [bs, 42]
        return out_qa, out_match_posi, out_match_nega

        # combined_feature = torch.mul(feat, question_avg)  # [bs, 512]
        # combined_feature = self.tanh(combined_feature)  # [bs, 512]
        # out_qa = self.fc_ans(combined_feature)  # [batch_size, ans_vocab_size] = [bs, 42]
        #
        # return out_qa, out_match_posi, out_match_nega


if __name__ == '__main__':
    audio = torch.randn(2, 10, 36, 1536)
    visual_posi = torch.randn(2, 10, 36, 1536)
    visual_nega = torch.randn(2, 10, 36, 1536)
    question = torch.randn(2, 77, 512)
    net = AVQA_Fusion_Net()
    out_qa, out_match_posi, out_match_nega = net(audio, visual_posi, visual_nega, question)
    print('out_qa:', out_qa.shape)
    print('out_match_posi:', out_match_posi.shape)
    print('out_match_nega:', out_match_nega.shape)
