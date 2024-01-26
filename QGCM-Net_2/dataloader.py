import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import pandas as pd
import ast
import json
from PIL import Image
# from munch import munchify
import time
import random

import warnings

warnings.filterwarnings('ignore')


def ids_to_multinomial(id, categories):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}
    return id_to_idx[id]


class AVQA_dataset(Dataset):
    def __init__(self, label, audio_dir, video_dir, question_dir,
                 transform=None, mode_flag='train'):

        samples = json.load(open('D:/CodeOfStudy/QGCM-Net-main/dataset/music_avqa_train.json', 'r'))

        # Question
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1
            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])
        # ques_vocab.append('fifth')

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.question_dir = question_dir
        self.transform = transform

        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list
        self.video_len = 60 * len(video_list)

        id_to_idx = {id: index for index, id in enumerate(self.ans_vocab)}
        with open('answers/answers.txt', 'w') as file:
            for key, value in id_to_idx.items():
                file.write(f'{key}:{value}\n')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample['video_id']
        question_id = sample['question_id']

        visual_posi = np.load(os.path.join(self.video_dir, name + '.npy'))
        visual_posi = visual_posi[:60:3, :]
        visual_posi = torch.from_numpy(visual_posi)

        video_idx = self.video_list.index(name)
        for i in range(visual_posi.shape[0]):
            while (1):
                neg_frame_id = random.randint(0, self.video_len - 1)
                if (int(neg_frame_id / 60) != video_idx):
                    break
            neg_video_id = int(neg_frame_id / 60)
            neg_frame_flag = neg_frame_id % 60
            neg_video_name = self.video_list[neg_video_id]
            visual_nega_out = np.load(os.path.join(self.video_dir, neg_video_name + '.npy'))
            visual_nega_out = torch.from_numpy(visual_nega_out)
            visual_nega_clip = visual_nega_out[neg_frame_flag, :, :].unsqueeze(0)
            if (i == 0):
                visual_nega = visual_nega_clip
            else:
                visual_nega = torch.cat((visual_nega, visual_nega_clip), dim=0)

        audio_feat = np.load(os.path.join(self.audio_dir, name + '.npy'))
        audio_feat = audio_feat[::3, :]
        audio_feat = torch.from_numpy(audio_feat)

        question_feat = np.load(os.path.join(self.question_dir, str(question_id) + '.npy'))
        question_feat = torch.from_numpy(question_feat)
        question_feat = question_feat.squeeze(0)

        answer = sample['anser']
        answer_label = ids_to_multinomial(answer, self.ans_vocab)
        answer_label = torch.from_numpy(np.array(answer_label)).long()

        sample = {
            'video_name': name,
            'question_id': question_id,
            'visual_posi': visual_posi,
            'visual_nega': visual_nega,
            'audio_feat': audio_feat,
            'question_feat': question_feat,
            'answer_label': answer_label,
        }
        if self.transform:
            sample = self.transform(sample)
        # sample = ToTensor(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        video_name = sample['video_name']
        question_id = sample['question_id'],
        visual_posi = sample['visual_posi']
        visual_nega = sample['visual_nega']
        audio_feat = sample['audio_feat']
        question_feat = sample['question_feat']
        answer_label = sample['answer_label']

        return {'video_name': video_name,
                'question_id': question_id,
                'visual_posi': torch.from_numpy(visual_posi),
                'visual_nega': torch.from_numpy(visual_nega),
                'audio_feat': torch.from_numpy(audio_feat),
                'question_feat': torch.from_numpy(question_feat),
                'answer_label': answer_label}


if __name__ == "__main__":
    video_dir = 'MUSIC_QA/Datasets/swinv2_l_p4_w12_videos'
    audio_dir = 'MUSIC_QA/Datasets/swinv2_l_p4_w12_audios'
    question_dir = 'MUSIC_QA/Datasets/clip_word'
    label_dir = 'MUSIC_QA/AVQA/dataset/music_avqa_train.json'
    # dataloader = AVQA_dataset(label_dir, audio_dir, video_dir, question_dir)
    # lens = dataloader.__len__()
    # print(lens)
    # for i in range(lens):
    #     sample = dataloader.__getitem__(i)
    #     video_name, visual_posi, visual_nega, audio_feat, question_feat, answer_label = sample['video_name'], sample['visual_posi'], \
    #                                                                                     sample['visual_nega'], sample['audio_feat'], \
    #                                                                                     sample['question_feat'], sample['answer_label']
    #     print(visual_posi.shape)
    #     print(visual_nega.shape)
    #     print(audio_feat.shape)
    #     print(question_feat.shape)
    #     print(answer_label.shape)
    #     break

    train_dataset = AVQA_dataset(label=label_dir,
                                 audio_dir=audio_dir,
                                 video_dir=video_dir,
                                 question_dir=question_dir,
                                 mode_flag='train')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8,
                              pin_memory=True)
    for batch_idx, sample in enumerate(train_loader):
        video_name, visual_posi, visual_nega, audio_feat, question_feat, answer_label = sample['video_name'], sample[
            'visual_posi'], \
                                                                                        sample['visual_nega'], sample[
                                                                                            'audio_feat'], \
                                                                                        sample['question_feat'], sample[
                                                                                            'answer_label']
        print(visual_posi.shape)
        print(visual_nega.shape)
        print(audio_feat.shape)
        print(question_feat.shape)
        print(answer_label.shape)
        break
