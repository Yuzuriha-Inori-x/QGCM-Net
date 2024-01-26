from __future__ import print_function
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import *
from net_grd_avst.net_avst import AVQA_Fusion_Net
# from net_grd_avst.net_avst3 import AVQA_Fusion_Net
# from net_grd_avst.net_avst_wrqst import AVQA_Fusion_Net
# from net_grd_avst.net_avst_wrav import AVQA_Fusion_Net
# from net_grd_avst.net_avst_wralg import AVQA_Fusion_Net
# from net_grd_avst.net_avst_wrall import AVQA_Fusion_Net

import ast
import json
import numpy as np

import warnings

# from datetime import datetime

# TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/net_avst/' + TIMESTAMP)

print("\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batch_organize(out_match_posi, out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    # print("audio data: ", audio_data.shape)
    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0

    return out_match, batch_labels


def model_train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in enumerate(train_loader):
        visual_posi, visual_nega, audio_feat, question_feat, target = sample['visual_posi'].to('cuda'), \
                                                                      sample['visual_nega'].to('cuda'), sample[
                                                                          'audio_feat'].to('cuda'), \
                                                                      sample['question_feat'].to('cuda'), sample[
                                                                          'answer_label'].to('cuda')

        optimizer.zero_grad()
        out_qa, out_match_posi, out_match_nega = model(audio_feat, visual_posi, visual_nega, question_feat)
        out_match, match_label = batch_organize(out_match_posi, out_match_nega)
        out_match, match_label = out_match.type(torch.FloatTensor).cuda(), match_label.type(torch.LongTensor).cuda()

        # output.clamp_(min=1e-7, max=1 - 1e-7)
        loss_match = criterion(out_match, match_label)
        loss_qa = criterion(out_qa, target)
        loss = loss_qa + 0.5 * loss_match

        # writer.add_scalar('run/match', loss_match.item(), epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('run/qa_test', loss_qa.item(), epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('run/both', loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio_feat), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def model_eval(model, val_loader):
    model.eval()
    total_qa = 0
    total_match = 0
    correct_qa = 0
    correct_match = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            visual_posi, visual_nega, audio_feat, question_feat, target = sample['visual_posi'].to('cuda'), \
                                                                          sample['visual_nega'].to('cuda'), sample[
                                                                              'audio_feat'].to('cuda'), \
                                                                          sample['question_feat'].to('cuda'), sample[
                                                                              'answer_label'].to('cuda')
            question_id = sample['question_id']
            preds_qa, out_match_posi, out_match_nega = model(audio_feat, visual_posi, visual_nega, question_feat)
            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()
            question_id = str(question_id.item())
            predicted_str = str(predicted.item())
            with open('answers/prediction_wtalg.txt', 'a') as file:
                file.write(f'{question_id}:{predicted_str}\n')
    print('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
    # writer.add_scalar('metric/acc_qa', 100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa


def model_test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    correct_top_01 = 0
    correct_top_02 = 0
    correct_top_03 = 0
    correct_top_04 = 0
    correct_top_05 = 0
    correct_top_06 = 0
    correct_top_07 = 0
    correct_top_08 = 0
    correct_top_09 = 0
    correct_top_10 = 0

    samples = json.load(open(r'D:\CodeOfStudy\QGCM-Net-main\dataset\music_avqa_test.json', 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            visual_posi, visual_nega, audio_feat, question_feat, target = sample['visual_posi'].to('cuda'), \
                                                                          sample['visual_nega'].to('cuda'), sample[
                                                                              'audio_feat'].to('cuda'), \
                                                                          sample['question_feat'].to('cuda'), sample[
                                                                              'answer_label'].to('cuda')

            preds_qa, out_match_posi, out_match_nega = model(audio_feat, visual_posi, visual_nega, question_feat)
            preds = preds_qa
            total += preds_qa.size(0)

            # top-01 accuracy
            _, predicted_top_01 = torch.max(preds_qa.data, 1)
            correct_top_01 += (predicted_top_01 == target).sum().item()

            # top-05 and top-20 accuracy
            _, predicted_top_n = torch.sort(preds_qa.data, dim=1, descending=True)

            predicted_top_02 = predicted_top_n[:, :2].detach().cpu().numpy()
            predicted_top_03 = predicted_top_n[:, :3].detach().cpu().numpy()
            predicted_top_04 = predicted_top_n[:, :4].detach().cpu().numpy()
            predicted_top_05 = predicted_top_n[:, :5].detach().cpu().numpy()
            predicted_top_06 = predicted_top_n[:, :6].detach().cpu().numpy()
            predicted_top_07 = predicted_top_n[:, :7].detach().cpu().numpy()
            predicted_top_08 = predicted_top_n[:, :8].detach().cpu().numpy()
            predicted_top_09 = predicted_top_n[:, :9].detach().cpu().numpy()
            predicted_top_10 = predicted_top_n[:, :10].detach().cpu().numpy()

            ground_truth = target.detach().cpu().numpy()
            n_batch = ground_truth.shape[0]
            ground_truth = ground_truth.reshape(n_batch, 1)

            correct_top_02 += np.count_nonzero((predicted_top_02 - ground_truth) == 0)
            correct_top_03 += np.count_nonzero((predicted_top_03 - ground_truth) == 0)
            correct_top_04 += np.count_nonzero((predicted_top_04 - ground_truth) == 0)
            correct_top_05 += np.count_nonzero((predicted_top_05 - ground_truth) == 0)
            correct_top_06 += np.count_nonzero((predicted_top_06 - ground_truth) == 0)
            correct_top_07 += np.count_nonzero((predicted_top_07 - ground_truth) == 0)
            correct_top_08 += np.count_nonzero((predicted_top_08 - ground_truth) == 0)
            correct_top_09 += np.count_nonzero((predicted_top_09 - ground_truth) == 0)
            correct_top_10 += np.count_nonzero((predicted_top_10 - ground_truth) == 0)

        val_top_01 = 100 * correct_top_01 / total
        val_top_02 = 100 * correct_top_02 / total
        val_top_03 = 100 * correct_top_03 / total
        val_top_04 = 100 * correct_top_04 / total
        val_top_05 = 100 * correct_top_05 / total
        val_top_06 = 100 * correct_top_06 / total
        val_top_07 = 100 * correct_top_07 / total
        val_top_08 = 100 * correct_top_08 / total
        val_top_09 = 100 * correct_top_09 / total
        val_top_10 = 100 * correct_top_10 / total

        print("\nTop-01 Validation set accuracy = %.2f %%" % val_top_01)
        print("Top-02 Validation set accuracy = %.2f %%" % val_top_02)
        print("Top-03 Validation set accuracy = %.2f %%" % val_top_03)
        print("Top-04 Validation set accuracy = %.2f %%" % val_top_04)
        print("Top-05 Validation set accuracy = %.2f %%" % val_top_05)
        print("Top-06 Validation set accuracy = %.2f %%" % val_top_06)
        print("Top-07 Validation set accuracy = %.2f %%" % val_top_07)
        print("Top-08 Validation set accuracy = %.2f %%" % val_top_08)
        print("Top-09 Validation set accuracy = %.2f %%" % val_top_09)
        print("Top-10 Validation set accuracy = %.2f %%" % val_top_10)

        return val_top_01
    #         _, predicted = torch.max(preds.data, 1)
    #
    #         total += preds.size(0)
    #         correct += (predicted == target).sum().item()
    #
    #         x = samples[batch_idx]
    #         type = ast.literal_eval(x['type'])
    #         if type[0] == 'Audio':
    #             if type[1] == 'Counting':
    #                 A_count.append((predicted == target).sum().item())
    #             elif type[1] == 'Comparative':
    #                 A_cmp.append((predicted == target).sum().item())
    #         elif type[0] == 'Visual':
    #             if type[1] == 'Counting':
    #                 V_count.append((predicted == target).sum().item())
    #             elif type[1] == 'Location':
    #                 V_loc.append((predicted == target).sum().item())
    #         elif type[0] == 'Audio-Visual':
    #             if type[1] == 'Existential':
    #                 AV_ext.append((predicted == target).sum().item())
    #             elif type[1] == 'Counting':
    #                 AV_count.append((predicted == target).sum().item())
    #             elif type[1] == 'Location':
    #                 AV_loc.append((predicted == target).sum().item())
    #             elif type[1] == 'Comparative':
    #                 AV_cmp.append((predicted == target).sum().item())
    #             elif type[1] == 'Temporal':
    #                 AV_temp.append((predicted == target).sum().item())
    #
    # print('Audio Counting Accuracy: %.2f %%' % (
    #         100 * sum(A_count) / len(A_count)))
    # print('Audio Cmp Accuracy: %.2f %%' % (
    #         100 * sum(A_cmp) / len(A_cmp)))
    # print('Audio Accuracy: %.2f %%' % (
    #         100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    # print('Visual Counting Accuracy: %.2f %%' % (
    #         100 * sum(V_count) / len(V_count)))
    # print('Visual Loc Accuracy: %.2f %%' % (
    #         100 * sum(V_loc) / len(V_loc)))
    # print('Visual Accuracy: %.2f %%' % (
    #         100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    # print('AV Ext Accuracy: %.2f %%' % (
    #         100 * sum(AV_ext) / len(AV_ext)))
    # print('AV counting Accuracy: %.2f %%' % (
    #         100 * sum(AV_count) / len(AV_count)))
    # print('AV Loc Accuracy: %.2f %%' % (
    #         100 * sum(AV_loc) / len(AV_loc)))
    # print('AV Cmp Accuracy: %.2f %%' % (
    #         100 * sum(AV_cmp) / len(AV_cmp)))
    # print('AV Temporal Accuracy: %.2f %%' % (
    #         100 * sum(AV_temp) / len(AV_temp)))
    #
    # print('AV Accuracy: %.2f %%' % (
    #         100 * (sum(AV_count) + sum(AV_loc) + sum(AV_ext) + sum(AV_temp)
    #                + sum(AV_cmp)) / (len(AV_count) + len(AV_loc) + len(AV_ext) + len(AV_temp) + len(AV_cmp))))
    #
    # print('Overall Accuracy: %.2f %%' % (
    #         100 * correct / total))
    #
    # return 100 * correct / total


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default=r'E:\DataSet\MUSIC-QA\swinv2_l_p4_w12_audios_v2', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default=r'E:\DataSet\MUSIC-QA\swinv2_l_p4_w12_videos_v2', help="video dir")
    parser.add_argument(
        "--question_dir", type=str, default=r'E:\DataSet\MUSIC-QA\clip_word', help="video dir")

    parser.add_argument(
        "--label_train", type=str, default=r"D:\CodeOfStudy\QGCM-Net-main\dataset\music_avqa_train.json",
        help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default=r"D:\CodeOfStudy\QGCM-Net-main\dataset\music_avqa_val.json",
        help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default=r"D:\CodeOfStudy\QGCM-Net-main\dataset\music_avqa_test.json",
        help="test csv file")

    parser.add_argument(
        '--early_stop', type=int, default=5, help='weight and bias setup')
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=2e-4, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='test', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='D:/CodeOfStudy/QGCM-Net-main/net_grd_avst/avst_models/',
        help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='wtall', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    set_seed(args.seed)

    if args.model == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net()
        # model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(label=args.label_train,
                                     audio_dir=args.audio_dir,
                                     video_dir=args.video_dir,
                                     question_dir=args.question_dir,
                                     mode_flag='train')

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True)

        val_dataset = AVQA_dataset(label=args.label_val,
                                   audio_dir=args.audio_dir,
                                   video_dir=args.video_dir,
                                   question_dir=args.question_dir,
                                   mode_flag='val')

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        counts = 0
        for epoch in range(1, args.epochs + 1):
            model_train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = model_eval(model, val_loader)
            counts += 1
            if F >= best_F:
                counts = 0
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
            if counts == args.early_stop:
                exit()
    elif args.mode == 'val':
        val_dataset = AVQA_dataset(label=args.label_val,
                                   audio_dir=args.audio_dir,
                                   video_dir=args.video_dir,
                                   question_dir=args.question_dir,
                                   mode_flag='val')

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        _ = model_eval(model, val_loader)

    else:
        test_dataset = AVQA_dataset(label=args.label_test,
                                    audio_dir=args.audio_dir,
                                    video_dir=args.video_dir,
                                    question_dir=args.question_dir,
                                    mode_flag='test')
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        model_test(model, test_loader)


if __name__ == '__main__':
    # main()

    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default=r'E:\DataSet\MUSIC-QA\swinv2_l_p4_w12_audios_v2', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default=r'E:\DataSet\MUSIC-QA\swinv2_l_p4_w12_videos_v2', help="video dir")
    parser.add_argument(
        "--question_dir", type=str, default=r'E:\DataSet\MUSIC-QA\clip_word', help="video dir")

    parser.add_argument(
        "--label_train", type=str, default=r"D:\CodeOfStudy\QGCM-Net-main\dataset\music_avqa_train.json",
        help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default=r"D:\CodeOfStudy\QGCM-Net-main\dataset\music_avqa_val.json",
        help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default=r"D:\CodeOfStudy\QGCM-Net-main\dataset\music_avqa_test.json",
        help="test csv file")

    parser.add_argument(
        '--early_stop', type=int, default=5, help='weight and bias setup')
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=2e-4, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='test', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='D:/CodeOfStudy/QGCM-Net-main/net_grd_avst/avst_models/',
        help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='token_64+lr_2e-4', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    set_seed(args.seed)

    if args.model == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net()
        # model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(label=args.label_train,
                                     audio_dir=args.audio_dir,
                                     video_dir=args.video_dir,
                                     question_dir=args.question_dir,
                                     mode_flag='train')

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)

        val_dataset = AVQA_dataset(label=args.label_val,
                                   audio_dir=args.audio_dir,
                                   video_dir=args.video_dir,
                                   question_dir=args.question_dir,
                                   mode_flag='val')

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        counts = 0
        for epoch in range(1, args.epochs + 1):
            model_train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = model_eval(model, val_loader)
            counts += 1
            if F >= best_F:
                counts = 0
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
            if counts == args.early_stop:
                exit()
    elif args.mode == 'val':
        val_dataset = AVQA_dataset(label=args.label_val,
                                   audio_dir=args.audio_dir,
                                   video_dir=args.video_dir,
                                   question_dir=args.question_dir,
                                   mode_flag='val')

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        _ = model_eval(model, val_loader)

    else:
        test_dataset = AVQA_dataset(label=args.label_test,
                                    audio_dir=args.audio_dir,
                                    video_dir=args.video_dir,
                                    question_dir=args.question_dir,
                                    mode_flag='test')
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        model_test(model, test_loader)
