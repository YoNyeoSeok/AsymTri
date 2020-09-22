from packaging import version
import argparse
import time
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
from eval_iou import (load_gt, label_mapping,
                      compute_threshold_table,
                      per_class_filter_two_output,
                      compute_filtered_output_a_b_and_or,
                      fast_hist_torch, per_class_iu_torch,
                      fast_hist_batch, per_class_iu_batch)

from model.deeplab_multi import DeeplabMulti
from model.deeplab_dualco import DeeplabDualCo
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

import wandb

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434),
#                     dtype=np.float32)
IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32)  # BGR
IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32)  # BGR
MODEL = 'DeepLabDualCo'
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

LEARNING_RATE = 2.5e-4  # 2.5e-4, 5e-5
MOMENTUM = 0.9
NUM_CLASSES = 19
# source: 24966, target: 2975
NUM_STEPS = 250000  # 250000, 30000
NUM_STEPS_START = 1
NUM_STEPS_STOP = 150000  # 150000, 18000  # early stopping (6 epoch)
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
# RESTORE_FROM = './model/gta_src.pth'
RESTORE_FROM_PSLABEL = '9ab2hb8f/GTA5_70000.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100  # 5000, 600
SNAPSHOT_DIR = './snapshots_diff_dualco_noisy/'
WEIGHT_DECAY = 0.001

WEIGHT_DIFF_ORTH = []
WEIGHT_DIFF_NORM = 2
LAMBDA_DIFF = [1., 1.]

PSEUDO_LABEL_THRESHOLD = '10,90_100'
PSEUDO_LABEL_POLICY = 'per_class_1and2'

CLEAN_SAMPLE_THRESHOLD = '10,90_100'
CLEAN_SAMPLE_POLICY = 'JoCoR_34'
LAMBDA_CLEAN_SAMPLE = [[1., 1.],
                       [1., 1.], ]
LAMBDA_CLEAN_SAMPLE_REG = [[1., 1.],
                           [1., 1.], ]
LAMBDA_CLEAN_SAMPLE_SEL = [[-1, 1],
                           [1, -1], ]

LAMBDA_TARGET = [1., 1., 1., 1.]

TARGET = 'cityscapes'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab, DeepLabDualCo")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
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

    parser.add_argument('--orth', type=int, nargs='*', default=WEIGHT_DIFF_ORTH, choices=[1, 2, 3],
                        help="orthogonal dimension for weight difference. [1] for channel, [2, 3] for kernel, [1, 2, 3] for whole feature")
    parser.add_argument('--norm', type=int, choices=[1, 2], default=WEIGHT_DIFF_NORM,
                        help="norm for weight difference.")
    parser.add_argument('--lambda_diff', type=float, nargs=2, default=LAMBDA_DIFF,
                        help="lambda_diff for weight difference.")
    parser.add_argument('--pslabel-threshold', type=str, default=PSEUDO_LABEL_THRESHOLD,
                        help="pseudo label threshold. ('10,00_100' ~ '10,90_100')")
    parser.add_argument('--pslabel-policy', type=str, default=PSEUDO_LABEL_POLICY,
                        help="pslabel policy (1, 2, 1and2, 1or2, per_class_1, per_class_2, per_class_1and2, per_class_1or2), default: per_class_1and2.")
    parser.add_argument('--clsample-threshold', type=str, default=CLEAN_SAMPLE_THRESHOLD,
                        help="clean sample threshold. ('10,00_100' ~ '10,90_100')")
    parser.add_argument('--clsample-policy', type=str, default=CLEAN_SAMPLE_POLICY,
                        help="clean label policy (DeCouple_34, CoTeaching_(plus_)(cls_)34, JoCoR_(plus_)(cls_)34), default: JoCoR_34.")
    parser.add_argument('--lambda-clean-sample', action='append', nargs='+', type=float, default=LAMBDA_CLEAN_SAMPLE,
                        help="param for clean sample policy")
    parser.add_argument('--lambda_target', type=float, nargs=4, default=LAMBDA_TARGET,
                        help="lambda_target for classifier losses of target domain.")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-steps-start", type=int, default=NUM_STEPS_START,
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
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-pslabel", type=str, default=RESTORE_FROM_PSLABEL,
                        help="Where restore pslabel model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--gpu-eval", type=int, default=1,
                        help="choose gpu device for eval.")
    parser.add_argument('--use-wandb', action='store_true')
    return parser.parse_args()


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


# def save_target_pred_max_argmax(model, targetloader, interp_target_gt, save_dir):
#     model.eval()
#     with torch.no_grad():
#         for ind, batch in enumerate(targetloader):
#             images, _, name = batch
#             images = images.cuda(args.gpu)

#             # 4xBxCxWxH
#             output1234 = torch.cat(model(images))
#             # 4xBxCxWxH
#             output1234 = interp_target_gt(output1234)
#             output1234 = output1234.reshape(4, -1, *output1234.shape[1:])
#             for n, o1234 in zip(name, output1234.split(1, dim=1)):
#                 # [4xCxWxH].max(1) -> (4xWxH), (4xWxH)
#                 max_argmax_o1234 = torch.stack(o1234.max(1))
#                 save_name = osp.splitext(n)[0]+'.pth'
#                 torch.save(max_argmax_o1234, osp.join(save_name, save_name))

def eval_batch(model, num_classes, batch, mapping, interp_target_gt, gpu):
    model.cuda(gpu)
    with torch.no_grad():
        images, _, name = batch
        images = images.cuda(gpu)
        gt_labelIds = load_gt(name, args.gt_dir_target,
                              'val').cuda(gpu)
        gt_trainIds = label_mapping(gt_labelIds, mapping)

        # 4BxCxWxH
        output1234 = torch.cat(model(images))
        # 4xBxCxWxH
        poten1234 = interp_target_gt(output1234)
        poten1234 = poten1234.reshape(4, -1, *poten1234.shape[1:])
        prob1234 = F.softmax(poten1234, dim=-3)
        # 4xBxWxH
        confid1234, pred1234 = prob1234.max(2)

        t = time.time()
        # 16xNxBxWxH
        filtered_pred_1_2_1and2_1or2_3_4_3and4_3or4 = torch.cat(list(map(
            lambda max_output_ab, argmax_output_ab:
                # 8xNxBxWxH
                torch.cat(list(map(
                    # 4xNxBxWxH
                    lambda igc: compute_filtered_output_a_b_and_or(
                        max_output_ab, argmax_output_ab,
                        ignore_label=args.ignore_label, C=num_classes, N=10, ignore_class=igc),
                    [True, False]
                ))),
            confid1234.split(2), pred1234.split(2)
        )))
        at = time.time()-t

        t = time.time()
        # WHY BATCH IS MORE SLOWER?
        # [16xNxBx19x19].sum(2) -> 16xNx19x19
        hist = torch.stack(list(map(
            lambda pred_batch: torch.stack(list(map(
                lambda gt, pred_batch: fast_hist_torch(
                    gt, pred_batch, num_classes),
                gt_trainIds, pred_batch
            ))),
            filtered_pred_1_2_1and2_1or2_3_4_3and4_3or4.flatten(0, -4)
        ))).reshape(*filtered_pred_1_2_1and2_1or2_3_4_3and4_3or4.shape[:-2], num_classes, num_classes).sum(2)
        # WHY BATCH IS MORE SLOWER?
        # [16xNxBx19x19].sum(2) -> 16xNx19x19
        # hist = fast_hist_batch(
        #     # [BxWxH].flatten(-2) -> BxWH
        #     gt_trainIds.flatten(-2),
        #     # [16xNxBxWxH].flatten(-2) -> 16xNxBxWH
        #     filtered_pred_1_2_1and2_1or2_3_4_3and4_3or4.flatten(
        #         -2),
        #     num_classes).sum(2)
        bt = time.time()-t
    return hist


def main(args):
    """Create the model and start the training."""

    if args.use_wandb:
        wandb.init(project='AsymTri_Pipeline_Diff_DualCo_Noisy2',
                   name='Round1', dir='./', config=args)

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
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = torch.from_numpy(
        np.array(info['label2train'])).byte().cuda(args.gpu)

    # Create network
    if 'DeepLab' in args.model:
        if args.model == 'DeepLabDualCo':
            model = DeeplabDualCo(num_classes=args.num_classes)
            prev_model = DeeplabDualCo(num_classes=args.num_classes)
        else:
            raise NotImplementedError(
                "wrong model name: {}".format(args.model))
        if args.is_training:
            model.train()
        else:
            model.eval()
        model.cuda(args.gpu)
        prev_model.eval()
        prev_model.cuda(args.gpu_eval)

        if args.restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(
                args.restore_from, map_location=torch.device(args.gpu))

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                # print i_parts
                if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                    # print i_parts
            saved_state_dict = new_params
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
        del saved_state_dict
        prev_state_dict = torch.load(
            args.restore_from_pslabel, map_location=torch.device(args.gpu))
        prev_state_dict.update([
            (k, v) for k, v in model.state_dict().items() if k not in prev_state_dict])
        prev_model.load_state_dict(prev_state_dict)
        del prev_state_dict
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

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean_rgb=IMG_MEAN[::-1], std_rgb=IMG_STD[::-1],
                                                     set='train'),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   #    pin_memory=True
                                   )

    targetloader_iter = enumerate(targetloader)

    testtargetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target_test,
                                                         crop_size=input_size_target,
                                                         scale=False, mirror=False, mean_rgb=IMG_MEAN[::-1], std_rgb=IMG_STD[::-1],
                                                         set='val'),
                                       batch_size=args.batch_size*1, shuffle=False,
                                       #    pin_memory=True
                                       )
    testtargetloader_iter = enumerate(testtargetloader)

    interp = nn.Upsample(
        size=(input_size[1], input_size[0]),
        mode='bilinear', align_corners=False)
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode='bilinear', align_corners=False)
    interp_target_gt = nn.Upsample(
        size=(input_size_target_gt[1], input_size_target_gt[0]),
        mode='bilinear', align_corners=False)

    n_threshold, threshold = args.pslabel_threshold.split(',')
    n_threshold = int(n_threshold)
    thr_columns = ['{0:02d}_{1:d}'.format(p, 100)
                   for p in range(0, 100, n_threshold)]
    policy_index = ['1', '2', '1and2', '1or2',
                    'per_class_1', 'per_class_2', 'per_class_1and2', 'per_class_1or2',
                    '3', '4', '3and4', '3or4',
                    'per_class_3', 'per_class_4', 'per_class_3and4', 'per_class_3or4',
                    ]
    clsample_thld_n, clsample_thld = args.clsample_threshold.split(',')
    clsample_thld_n = int(clsample_thld_n)

    params = model.optim_parameters(args)
    optimizer = optim.SGD(params,
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    eval_hist = torch.zeros(
        (len(testtargetloader), len(policy_index), len(thr_columns), num_classes, num_classes)).cuda(args.gpu_eval)

    for i_iter in range(args.num_steps_start, args.num_steps):
        if args.is_training:
            model.train()
        else:
            model.eval()
        model.cuda(args.gpu)
        prev_model.eval()
        prev_model.cuda(args.gpu_eval)

        loss_seg_value1 = 0
        loss_seg_value2 = 0
        loss_diff_value = 0
        loss_seg_target_value1 = 0
        loss_seg_target_value2 = 0
        loss_seg_target_value3 = 0
        loss_seg_target_value4 = 0

        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):
            # train with source
            optimizer.zero_grad()
            _, batch = next(trainloader_iter)
            images, labels, _, _ = batch
            images = images.cuda(args.gpu)
            labels = labels.long().cuda(args.gpu)

            # 4BxCxWxH
            output1234 = torch.cat(model(images))
            # 4BxCxWxH
            poten1234 = interp(output1234)
            # 4xBxCxWxH
            poten1234 = poten1234.reshape(4, -1, *poten1234.shape[1:])
            # 4xBxWxH

            loss_seg1, loss_seg2, _, _ = list(map(
                lambda poten: CrossEntropy2d(
                    ignore_label=args.ignore_label)(poten, labels),
                # lambda poten: loss_calc(
                #     poten, labels, args.ignore_label, args.gpu),
                poten1234))

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

            del batch, images, output1234, loss_seg1, loss_seg2, loss_diff, loss
            # torch.cuda.empty_cache()

            # train with target
            optimizer.zero_grad()
            _, batch = next(targetloader_iter)
            images, _, name = batch
            images = images.cuda(args.gpu_eval)

            # get pseudo-label
            prev_model.eval()
            with torch.no_grad():
                # 4BxCxWxH
                pm_output1234 = torch.cat(prev_model(images))
                # 4BxCxWxH
                pm_poten1234 = interp_target(pm_output1234)
                # 4xBxCxWxH
                pm_poten1234 = pm_poten1234.reshape(
                    4, -1, *pm_poten1234.shape[1:])
                pm_prob1234 = F.softmax(pm_poten1234, -3)
                # 4xBxWxH
                pm_confid1234, pm_pred1234 = pm_prob1234.max(2)

                # 16xNxBxWxH
                filtered_pm_pred_1_2_1and2_1or2_3_4_3and4_3or4 = torch.cat(list(map(
                    lambda max_output_ab, argmax_output_ab:
                        # 8xNxBxWxH
                        torch.cat(list(map(
                            # 4xNxBxWxH
                            lambda igc: compute_filtered_output_a_b_and_or(
                                max_output_ab, argmax_output_ab,
                                ignore_label=args.ignore_label, C=num_classes, N=10, ignore_class=igc),
                            [True, False]
                        ))),
                    pm_confid1234.split(2), pm_pred1234.split(2)
                )))
                # pslabel = pm_pred (prev model filtered pred)
                pslabel = filtered_pm_pred_1_2_1and2_1or2_3_4_3and4_3or4[
                    policy_index.index(args.pslabel_policy),
                    thr_columns.index(threshold)]
                del pm_poten1234, pm_prob1234, pm_confid1234, pm_pred1234
                del filtered_pm_pred_1_2_1and2_1or2_3_4_3and4_3or4

            # get clean label from pseudo-label
            model.eval()
            with torch.no_grad():
                output34 = torch.cat(model(images))[2:]
                # 4xBxCxWxH
                poten34 = interp_target(output34)
                poten34 = poten34.reshape(2, -1, *poten34.shape[1:])
                log_prob34 = F.log_softmax(poten34, dim=-3)
                prob34 = F.softmax(poten34, dim=-3)

                loss_seg_target34 = list(map(
                    lambda poten: nn.CrossEntropyLoss(
                        reduction='none', ignore_index=args.ignore_label)(poten, pslabel),
                    # lambda pred_target: loss_calc(
                    #     pred_target, pslabel, args.ignore_label, args.gpu),
                    poten34))

                if 'JoCoR' in args.clsample_policy:
                    # KLDivLoss(x,y) = y * [log(y) - log(x)] = KL[y|x]
                    kldiv_loss = torch.stack([
                        nn.KLDivLoss(reduction='none')(lp1, p2)
                        if e1 != e2 else
                        nn.KLDivLoss(reduction='none')(
                            lp1, (pslabel[:, None] == torch.arange(num_classes)[None, :, None, None].to(args.gpu)).float())
                        for e1, lp1 in enumerate(log_prob34) for e2, p2 in enumerate(prob34)
                    ]).reshape(2, 2, *prob34.shape[1:])
                    loss_mat = kldiv_loss * torch.from_numpy(np.array(
                        args.lambda_clean_sample)).float().cuda(args.gpu)[:, :, None, None, None, None]
                    loss34 = loss_mat.sum(1).sum(2, keepdim=True)
                    del kldiv_loss, loss_mat
                elif 'CoTeaching' in args.clsample_policy:
                    loss34 = torch.stack(loss_seg_target34)
                else:
                    raise NotImplementedError
                # loss34

                if 'plus_cls' in args.clsample_policy:
                    # for per class
                    loss34 = loss34.repeat(1, 1, num_classes, 1, 1)
                    mask34 = (pslabel != args.ignore_label)[None, :, None, :, :] \
                        + (prob34.argmax(2, keepdim=True) ==
                            torch.arange(num_classes).to(args.gpu)[None, None, :, None, None])
                elif 'plus' in args.clsample_policy:
                    # ignore class
                    loss34 = loss34.squeeze(2)
                    mask34 = (pslabel != args.ignore_label)[
                        None, :].repeat(2, 1, 1, 1)
                elif 'cls' in args.clsample_policy:
                    # for per class
                    loss34 = loss34.sum(0, keepdim=True).repeat(
                        2, 1, num_classes, 1, 1)
                    mask34 = (pslabel != args.ignore_label)[None, :, None, :, :] \
                        + (prob34.argmax(2, keepdim=True) ==
                            torch.arange(num_classes).to(args.gpu)[None, None, :, None, None])
                else:
                    # ignore class, ignore classifier
                    loss34 = loss34.squeeze(2).sum(
                        0, keepdim=True).repeat(2, 1, 1, 1)
                    mask34 = (pslabel != args.ignore_label)[
                        None, :].repeat(2, 1, 1, 1)
                # mask34

                masked_fill_loss34 = loss34.masked_fill(
                    ~mask34, float('-inf'))
                sorted_loss34 = masked_fill_loss34.flatten(
                    -2).sort(-1, descending=True)[0]
                mask_sum34 = mask34.flatten(
                    -2).sum(-1, keepdim=True).float()
                loss34_quantile = sorted_loss34.gather(
                    -1, (mask_sum34*0.5).round().long())

                if 'cls' not in args.clsample_policy:
                    pred34_filtered = loss34 < loss34_quantile[..., None]
                else:
                    loss34_thld_mask = loss34 < loss34_quantile[..., None]
                    pred34 = prob34.argmax(2)
                    pred34_filtered = pred34.masked_fill(
                        loss34_thld_mask.gather(
                            2, pred34.unsqueeze(2)).squeeze(2),
                        float(args.ignore_label))
                    del loss34_thld_mask, pred34

                selected_sample_3, selected_sample_4 = pred34_filtered

                if 'plus' in args.clsample_policy:
                    # minus
                    mat_selected_sample_and = torch.stack([
                        pf1.masked_fill(
                            pf1 != pf2, float(args.ignore_label))
                        for e1, pf1 in enumerate(pred34_filtered)
                        for e2, pf2 in enumerate(pred34_filtered)
                    ]).reshape(2, 2, *pred34_filtered.shape[1:])
                    mat_selected_sample_minus = torch.stack([
                        pf1.masked_fill(
                            pf1 == pf_and, float(args.ignore_label))
                        for e1, pf1 in enumerate(pred34_filtered)
                        for e2, pf_and in enumerate(mat_selected_sample_and[e1])
                    ]).reshape(2, 2, *pred34_filtered.shape[1:])

                    selected_sample_for3, selected_sample_for4 = \
                        (mat_selected_sample_minus !=
                            args.ignore_label).sum(dim=1).bool()
                    selected_samples = [
                        selected_sample_for3,
                        selected_sample_for4,
                    ]
                    del mat_selected_sample_and, mat_selected_sample_minus
                else:
                    selected_samples = [
                        selected_sample_3,
                        selected_sample_4,
                    ]

                del output34, poten34, log_prob34, prob34
                del loss34, mask34
                del masked_fill_loss34, sorted_loss34, mask_sum34, loss34_quantile
                del pred34_filtered, selected_sample_3, selected_sample_4

            # train with clean label
            if args.is_training:
                model.train()
            else:
                model.eval()

            images = images.cuda(args.gpu)
            pslabel = pslabel.cuda(args.gpu)
            output1234 = torch.cat(model(images))
            # 4xBxCxWxH
            poten1234 = interp_target(output1234)
            poten1234 = poten1234.reshape(4, -1, *poten1234.shape[1:])
            loss_seg_target1234 = list(map(
                lambda poten: nn.CrossEntropyLoss(
                    reduction='none', ignore_index=args.ignore_label)(poten, pslabel),
                poten1234))

            if 'JoCoR' in args.clsample_policy:
                log_prob34 = F.log_softmax(poten1234[2:], dim=-3)
                prob34 = F.softmax(poten1234[2:], dim=-3)
                # KLDivLoss(x,y) = y * [log(y) - log(x)] = KL[y|x]
                kldiv_loss = torch.stack([
                    nn.KLDivLoss(reduction='none')(lp1, p2)
                    if e1 != e2 else
                    nn.KLDivLoss(reduction='none')(
                        lp1, (pslabel[:, None] == torch.arange(num_classes)[None, :, None, None].to(args.gpu)).float())
                    for e1, lp1 in enumerate(log_prob34) for e2, p2 in enumerate(prob34)
                ]).reshape(2, 2, *prob34.shape[1:])
                loss_mat = kldiv_loss * torch.from_numpy(np.array(
                    args.lambda_clean_sample)).float().cuda(args.gpu)[:, :, None, None, None, None]
                loss34 = loss_mat.sum(1).sum(2)

                loss_seg_target1234[2:] = loss34
                del log_prob34, prob34, kldiv_loss, loss_mat, loss34

            loss_seg_target1234 = list(map(
                lambda a, l, s: a*(l if s else l[s]).mean(),
                args.lambda_target,
                loss_seg_target1234,
                [True, True] + selected_samples,
            ))
            del selected_samples

            loss = 0
            for l in loss_seg_target1234:
                if not torch.isnan(l):
                    loss += l

            loss = loss / args.iter_size
            loss.backward()

            loss_seg_target1, loss_seg_target2, loss_seg_target3, loss_seg_target4 = loss_seg_target1234
            loss_seg_target_value1 += loss_seg_target1.detach().cpu().numpy() / \
                args.iter_size
            loss_seg_target_value2 += loss_seg_target2.detach().cpu().numpy() / \
                args.iter_size
            loss_seg_target_value3 += loss_seg_target3.detach().cpu().numpy() / \
                args.iter_size
            loss_seg_target_value4 += loss_seg_target4.detach().cpu().numpy() / \
                args.iter_size

            optimizer.step()

            # train metric
            gt_labelIds = load_gt(name, args.gt_dir_target,
                                  'train').cuda(args.gpu_eval)
            gt_trainIds = label_mapping(gt_labelIds, mapping)

            with torch.no_grad():
                # 4BxCxWxH
                pm_poten1234 = interp_target_gt(pm_output1234)
                # 4xBxCxWxH
                pm_poten1234 = pm_poten1234.reshape(
                    4, -1, *pm_poten1234.shape[1:])
                # 4xBxWxH
                pred1234 = pm_poten1234.argmax(2)

                # hist = fast_hist_batch(
                #     gt_trainIds.flatten(-2), argmax_output1234.flatten(-2), num_classes)
                hist = torch.stack(list(map(
                    lambda pred_batch: torch.stack(list(map(
                        lambda gt, pred_batch: fast_hist_torch(
                            gt, pred_batch, num_classes),
                        gt_trainIds, pred_batch
                    ))),
                    pred1234.flatten(0, -4)
                ))).reshape(*pred1234.shape[:-2], num_classes, num_classes).sum(2)
                IoUs = per_class_iu_batch(hist).cpu().numpy()
                mIoU = np.nanmean(IoUs, axis=-1)
                del pm_poten1234, pred1234, hist

            print('  '.join(
                ['iter = {:8d}/{:8d}'.format(i_iter, args.num_steps),
                 '\n',
                 'train/loss_seg_source1 = {:.3f}'.format(loss_seg_value1),
                 'train/loss_seg_source2 = {:.3f}'.format(loss_seg_value2),
                 'train/loss_diff = {:.3f}'.format(loss_diff_value),
                 '\n',
                 'train/loss_seg_target1 = {:.3f}'.format(
                     loss_seg_target_value1),
                 'train/loss_seg_target2 = {:.3f}'.format(
                     loss_seg_target_value2),
                 'train/loss_seg_target3 = {:.3f}'.format(
                     loss_seg_target_value3),
                 'train/loss_seg_target4 = {:.3f}'.format(
                     loss_seg_target_value4),
                 '\n',
                 ] + ['train/mIoU_target{:d} = {:.3f}'.format(i+1, m) for i, m in enumerate(mIoU)]
            ))
            if args.use_wandb:
                wandb.log({
                    'train/loss_seg1': loss_seg_value1,
                    'train/loss_seg2': loss_seg_value2,
                    'train/loss_diff': loss_diff_value,
                    'train/loss_seg_target1': loss_seg_target_value1,
                    'train/loss_seg_target2': loss_seg_target_value2,
                    'train/loss_seg_target3': loss_seg_target_value3,
                    'train/loss_seg_target4': loss_seg_target_value4,
                    'train/mIoU_target1': mIoU[0],
                    'train/mIoU_target2': mIoU[1],
                    'train/mIoU_target3': mIoU[2],
                    'train/mIoU_target4': mIoU[3],
                }, step=i_iter)
                # wandb.log({
                #     'train/{}/IoU_target{}'.format(k, i+1): IoUs[i, idx] for idx, k in enumerate(name_classes) for i in range(4)
                # }, step=i_iter)
                # }, step=i_iter)

            # running eval
            with torch.no_grad():
                try:
                    _, batch = next(testtargetloader_iter)
                except StopIteration:
                    testtargetloader_iter = iter(testtargetloader)
                    _, batch = next(testtargetloader_iter)

                eval_hist[(i_iter-1) % len(testtargetloader)] = eval_batch(
                    model, num_classes, batch, mapping, interp_target_gt, args.gpu_eval)
            IoUs = per_class_iu_batch(
                eval_hist[:(i_iter-1)].sum(0)).cpu().numpy()
            mIoU = np.nanmean(IoUs, -1)

            dfs = {}
            for ind_class in range(num_classes):
                name_class = name_classes[ind_class]
                df = pd.DataFrame(IoUs[:, :, ind_class],
                                  index=policy_index, columns=thr_columns)
                dfs.update({name_class: df})
            df = pd.DataFrame(mIoU, index=policy_index, columns=thr_columns)
            dfs.update({'mIoU': df})
            print('  '.join(
                ['iter = {:8d}/{:8d}'.format(i_iter, args.num_steps),
                 '\n',
                 'eval/per_class_1and2_50_100_mIoU = {:.3f}'.format(
                     dfs['mIoU'].loc['per_class_1and2', '50_100']),
                 'eval/per_class_3_50_100_mIoU = {:.3f}'.format(
                     dfs['mIoU'].loc['per_class_3', '50_100']),
                 'eval/per_class_4_50_100_mIoU = {:.3f}'.format(
                     dfs['mIoU'].loc['per_class_4', '50_100']),
                 ]
            ))
            if args.use_wandb:
                for name_class, df in dfs.items():
                    wandb.log({
                        'eval/{}_{}_{}'.format(top_p, filter_output, name_class): df.loc[top_p, filter_output]
                        for top_p in policy_index
                        for filter_output in thr_columns
                    }, step=int(i_iter))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            save_dir = args.snapshot_dir if not args.use_wandb else wandb.run.dir
            if not os.path.exists(osp.join(save_dir, 'GTA5_' + str(args.num_steps_stop))):
                os.makedirs(osp.join(
                    save_dir, 'GTA5_' + str(args.num_steps_stop)))
            torch.save(model.state_dict(), osp.join(
                save_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            # dfs = eval_model(model, policy_index, thr_columns, threshold, num_classes, name_classes,
            #                  testtargetloader, mapping, interp_target_gt, args.gpu_eval)
            # if args.use_wandb:
            #     for name_class, df in pslabel_dfs.items():
            #         wandb.log({
            #             'pslabel/{}_{}_{}'.format(top_p, filter_output, name_class): df.loc[top_p, filter_output]
            #             for top_p in policy_index
            #             for filter_output in thr_columns
            #         }, step=int(args.num_steps_stop))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot % evaluating target domain ...')
            save_dir = args.snapshot_dir if not args.use_wandb else wandb.run.dir
            if not os.path.exists(osp.join(save_dir, 'GTA5_' + str(i_iter))):
                os.makedirs(osp.join(save_dir, 'GTA5_' + str(i_iter)))
            torch.save(model.state_dict(), osp.join(
                save_dir, 'GTA5_' + str(i_iter) + '.pth'))

            # dfs = eval_model(model, policy_index, thr_columns, threshold, num_classes, name_classes,
            #                  testtargetloader, mapping, interp_target_gt, args.gpu_eval)
            # if args.use_wandb:
            #     for name_class, df in pslabel_dfs.items():
            #         wandb.log({
            #             'eval/pslabel/{}_{}_{}'.format(top_p, filter_output, name_class): df.loc[top_p, filter_output]
            #             for top_p in policy_index
            #             for filter_output in thr_columns
            #         }, step=i_iter)
            # save_target_pred_max_argmax(model, targetloader, interp_target_gt,
            #                             osp.join(save_dir, 'GTA5_' + str(i_iter)))


if __name__ == '__main__':
    args = get_arguments()
    print(args)
    main(args)
