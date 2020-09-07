import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import json
import pandas as pd
from PIL import Image
from eval_iou import eval_iou

from model.deeplab_multi import DeeplabMulti
from model.deeplab_diff import DeeplabDiff
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

import wandb

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434),
#                     dtype=np.float32)
IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32)  # BGR
IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32)  # BGR
MODEL = 'DeepLabDiff'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 1  # 4

DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'

DATA_DIRECTORY_TARGET = './data/Cityscapes/leftImg8bit'
DEVKIT_DIR = './dataset/cityscapes_list/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
DATA_LIST_PATH_TARGET_TEST = './dataset/cityscapes_list/val.txt'
LOGIT_THRESHOLD_DIRECTORY_TARGET = './logit_threshold/cityscapes'
INPUT_SIZE_TARGET = '1024,512'

GT_DIRECTORY_TARGET = './data/Cityscapes/gtFine/'
GT_LIST_PATH_TARGET_TEST = './dataset/cityscapes_list/label.txt'
INPUT_SIZE_TARGET_GT = '2048,1024'

LEARNING_RATE = 5e-5  # 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
# source: 24966, target: 2975
NUM_STEPS = 30000  # 250000
NUM_STEPS_STOP = 18000  # 150000  # early stopping (6 epoch)
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM = './model/gta_src.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100  # 600  # 5000
SNAPSHOT_DIR = './snapshots_diff/'
WEIGHT_DECAY = 0.001

ORTH = []
NORM = 2
LAMBDA_DIFF = [0.5, 0.5]  # 0.005

TARGET = 'cityscapes'
# SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab, DeepLabDiff")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-target-test", type=str, default=DATA_LIST_PATH_TARGET_TEST,
                        help="Path to the file listing the images in the target test dataset.")
    parser.add_argument("--logit-threshold-dir-target", type=str, default=LOGIT_THRESHOLD_DIRECTORY_TARGET,
                        help="Path to the directory saving the logit and threshold of target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--gt-dir-target", type=str, default=GT_DIRECTORY_TARGET,
                        help="Path to the directory containing the target gt dataset.")
    parser.add_argument("--gt-list-target-test", type=str, default=GT_LIST_PATH_TARGET_TEST,
                        help="Path to the file listing the images in the target gt dataset.")
    parser.add_argument("--input-size-target-gt", type=str, default=INPUT_SIZE_TARGET_GT,
                        help="Comma-separated string with height and width of target gt images.")
    parser.add_argument('--orth', type=int, nargs='*', default=ORTH,
                        choices=[1, 2, 3],
                        help="orthogonal dimension for weight difference."
                             "[1] for channel, [2, 3] for kernel, [1, 2, 3] for whole feature")
    parser.add_argument('--norm', type=int, default=NORM,
                        choices=[1, 2],
                        help="norm for weight difference.")
    parser.add_argument('--lambda_diff', type=float, nargs=2, default=LAMBDA_DIFF,
                        help="lambda_diff for weight difference.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps-start", type=int, default=0,
                        help="Number of training steps to start.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    # parser.add_argument("--set", type=str, default=SET,
    #                     help="choose adaptation set.")
    parser.add_argument('--use-wandb', action='store_true')
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, ignore_label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(ignore_label=ignore_label).cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    if args.use_wandb:
        wandb.init(project='AsymTri_Pipeline_Diff',
                   name='SourceOnly', dir='./', config=args)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    w, h = map(int, args.input_size_target_gt.split(','))
    input_size_target_gt = (w, h)

    cudnn.enabled = True

    with open(osp.join(DEVKIT_DIR, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    assert num_classes == args.num_classes

    # Create network
    if 'DeepLab' in args.model:
        if args.model == 'DeepLabDiff':
            model = DeeplabDiff(num_classes=args.num_classes)
        else:
            raise NotImplementedError(
                "wrong model name: {}".format(args.model))

        model.train()
        model.cuda(args.gpu)

        if args.restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(
                args.restore_from, map_location=torch.device(args.gpu))
            raise NotImplementedError("TODO")
        elif args.restore_from.split('/')[-1] == 'gta_src.pth':
            saved_state_dict = torch.load(
                args.restore_from, map_location=torch.device(args.gpu))
            saved_state_dict.update([
                (k, v) for k, v in model.state_dict().items() if 'layer5' in k])
        else:
            saved_state_dict = torch.load(
                args.restore_from, map_location=torch.device(args.gpu))

        saved_state_dict.update([
            (k, v) for k, v in model.state_dict().items() if k not in saved_state_dict])
        model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(
            "wrong model name: {}".format(args.model))

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean_rgb=IMG_MEAN[::-1], std_rgb=IMG_STD[::-1]),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        # pin_memory=True,
    )

    trainloader_iter = enumerate(trainloader)

    testtargetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target_test,
                                                         crop_size=input_size_target,
                                                         scale=False, mirror=False, mean_rgb=IMG_MEAN[::-1], std_rgb=IMG_STD[::-1],
                                                         set='val'),
                                       batch_size=args.batch_size*10, shuffle=False,
                                       #    pin_memory=True
                                       )
    testgttargetloader = data.DataLoader(cityscapesDataSet(args.gt_dir_target, args.gt_list_target_test,
                                                           crop_size=input_size_target_gt,
                                                           scale=False, mirror=False, mean_rgb=IMG_MEAN[::-1], std_rgb=IMG_STD[::-1],
                                                           set='val'),
                                         batch_size=args.batch_size*10, shuffle=False,
                                         #  pin_memory=True
                                         )
    interp = nn.Upsample(
        size=(input_size[1], input_size[0]),
        mode='bilinear', align_corners=False)
    interp_target_gt = nn.Upsample(
        size=(input_size_target_gt[1], input_size_target_gt[0]),
        mode='bilinear', align_corners=False)

    index = ['{0:02d}_{1:d}'.format(p, 100)
             for p in range(0, 100, 10)]
    columns = ['1', '2', 'and', 'or', 'per_class_1',
               'per_class_2', 'per_class_and', 'per_class_or']
    dfs = eval_iou(info, model, testgttargetloader, testtargetloader,
                   interp_target_gt, args, index, columns)
    if args.use_wandb:
        for name_class, df in dfs.items():
            wandb.log({
                '{}_{}_{}'.format(top_p, filter_output, name_class): df.loc[top_p, filter_output]
                for top_p in index
                for filter_output in columns
            }, step=0)

    params = model.optim_parameters(args)
    optimizer = optim.SGD(params,
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    for i_iter in range(args.num_steps_start, args.num_steps):

        loss_seg_value1 = 0
        loss_seg_value2 = 0
        loss_diff_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):
            # train with source
            _, batch = next(trainloader_iter)
            images, labels, _, _ = batch
            images = Variable(images).cuda(args.gpu)

            pred12 = model(images)
            pred12 = list(map(interp, pred12))

            loss_seg12 = list(map(
                lambda pred: loss_calc(
                    pred, labels, args.ignore_label, args.gpu),
                pred12))
            loss_seg1, loss_seg2 = loss_seg12

            loss_diff = model.weight_diff(
                orth=args.orth, norm=args.norm)
            loss = args.lambda_diff[0] * (loss_seg1 + loss_seg2) + \
                args.lambda_diff[1] * loss_diff

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.detach().cpu().numpy() / args.iter_size
            loss_seg_value2 += loss_seg2.detach().cpu().numpy() / args.iter_size
            loss_diff_value += loss_diff.detach().cpu().numpy() / args.iter_size

        optimizer.step()
        print(' \t '.join([
            'iter = {:8d}/{:8d}'.format(i_iter, args.num_steps),
            'loss_seg1 = {:.3f}'.format(loss_seg_value1),
            'loss_seg2 = {:.3f}'.format(loss_seg_value2),
            'loss_diff = {:.3f}'.format(loss_diff_value),
        ]))
        if args.use_wandb:
            wandb.log({
                'loss_seg1': loss_seg_value1,
                'loss_seg2': loss_seg_value2,
                'loss_diff': loss_diff_value,
            }, step=i_iter)

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            save_dir = args.snapshot_dir if not args.use_wandb else wandb.run.dir
            if not os.path.exists(osp.join(save_dir, 'GTA5_' + str(args.num_steps_stop))):
                os.makedirs(osp.join(
                    save_dir, 'GTA5_' + str(args.num_steps_stop)))
            torch.save(model.state_dict(), osp.join(
                save_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))

            dfs = eval_iou(info, model, testgttargetloader, testtargetloader,
                           interp_target_gt, args, index, columns)
            for name_class, df in dfs.items():
                if args.use_wandb:
                    wandb.log({
                        '{}_{}_{}'.format(top_p, filter_output, name_class): df.loc[top_p, filter_output]
                        for top_p in index
                        for filter_output in columns
                    }, step=args.num_steps_stop)
                df.to_csv(osp.join(save_dir, 'GTA5_' + str(args.num_steps_stop) + '{}.csv'.format(name_class)),
                          sep='\t')
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot % evaluating target domain ...')
            save_dir = args.snapshot_dir if not args.use_wandb else wandb.run.dir
            if not os.path.exists(osp.join(save_dir, 'GTA5_' + str(i_iter))):
                os.makedirs(osp.join(save_dir, 'GTA5_' + str(i_iter)))
            torch.save(model.state_dict(), osp.join(
                save_dir, 'GTA5_' + str(i_iter) + '.pth'))

            dfs = eval_iou(info, model, testgttargetloader, testtargetloader,
                           interp_target_gt, args, index, columns,
                           print_index=-1, print_columns=4)
            for name_class, df in dfs.items():
                if args.use_wandb:
                    wandb.log({
                        '{}_{}_{}'.format(top_p, filter_output, name_class): df.loc[top_p, filter_output]
                        for top_p in index
                        for filter_output in columns
                    }, step=i_iter)
                df.to_csv(osp.join(save_dir, 'GTA5_' + str(i_iter) + '{}.csv'.format(name_class)),
                          sep='\t')


if __name__ == '__main__':
    main()
