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

from model.deeplab_multi import DeeplabMulti
from model.deeplab_dualco import DeeplabDualCo
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

import wandb

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLabDualCo'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 1 #4
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots_dualco/'
WEIGHT_DECAY = 0.001
ORTH = []
NORM = 2
LAMBDA_DIFF = [0.5, 0.5]  # 0.005

TARGET = 'cityscapes'
SET = 'train'


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
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument('--orth', type=int, nargs='*', default=ORTH,
                        choices=[1, 2, 3],
                        help="orthogonal dimension for weight difference."
                             "[1] for channel, [2, 3] for kernel, [1, 2, 3] for whole feature")
    parser.add_argument('--norm', type=int, default=NORM,
                        choices=[1, 2],
                        help="norm for weight difference.")

    parser.add_argument('--lambda_diff', type=float, nargs=2, default=LAMBDA_DIFF,
                        help="lambda_diff for weight difference.")

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
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument('--use-wandb', action='store_true')
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

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
        wandb.init(project='AsymTri_DualCo', name='SourceOnly', dir='./', config=args)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if 'DeepLab' in args.model:
        if args.model == 'DeepLab':
            model = DeeplabMulti(num_classes=args.num_classes)
        elif args.model == 'DeepLabDualCo':
            model = DeeplabDualCo(num_classes=args.num_classes)

        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
            i_name = 1
        else:
            saved_state_dict = torch.load(args.restore_from)
            saved_state_dict.update([(k, v) for k, v in model.state_dict().items() if 'layer7' in k or 'layer8' in k])
            model.load_state_dict(saved_state_dict)
            i_name = 0

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # for i_iter in range(args.num_steps_start, args.num_steps):

    #     loss_seg_value1 = 0
    #     loss_seg_value2 = 0

    #     optimizer.zero_grad()
    #     adjust_learning_rate(optimizer, i_iter)

    #     for sub_i in range(args.iter_size):
    #         # train with source

    #         _, batch = next(trainloader_iter)
    #         images, labels, _, _ = batch
    #         images = Variable(images).cuda(args.gpu)

    #         pred1234 = model(images)
    #         pred1, pred2, pred3, pred4 = list(map(interp, pred1234))

    #         loss_seg1 = loss_calc(pred1, labels, args.gpu)
    #         loss_seg2 = loss_calc(pred2, labels, args.gpu)
    #         loss_weight_diff = model.weight_diff()
    #         loss = loss_seg1 + loss_seg2 + args.lambda_diff * loss_weight_diff

    #         # proper normalization
    #         loss = loss / args.iter_size
    #         loss.backward()
    #         loss_seg_value1 += loss_seg1.detach().cpu().numpy() / args.iter_size
    #         loss_seg_value2 += loss_seg2.detach().cpu().numpy() / args.iter_size

    #         # train with target

    #         # _, batch = next(targetloader_iter)
    #         # images, _, _ = batch
    #         # images = Variable(images).cuda(args.gpu)

    #         # # pred_target1, pred_target2 = model(images)
    #         # _, _, pred_target = model(images)
    #         # # pred_target1 = interp_target(pred_target1)
    #         # # pred_target2 = interp_target(pred_target2)
    #         # pred_target = interp_target(pred_target)

    #         # # D_out1 = model_D1(F.softmax(pred_target1))
    #         # # D_out2 = model_D2(F.softmax(pred_target2))

    #         # # loss_adv_target1 = bce_loss(D_out1,
    #         # #                            Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(
    #         # #                                args.gpu))

    #         # # loss_adv_target2 = bce_loss(D_out2,
    #         # #                             Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
    #         # #                                 args.gpu))

    #         # # loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
    #         # # loss = loss / args.iter_size
    #         # # loss.backward()
    #         # # loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy()[0] / args.iter_size
    #         # # loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy()[0] / args.iter_size


    #     optimizer.step()
    #     print('iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_weight_diff = {4:.3f}'.format(
    #         i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_weight_diff))
    #     if args.use_wandb:
    #         wandb.log({'loss_seg1': loss_seg_value1, 'loss_seg2': loss_seg_value2, 'loss_weight_diff': loss_weight_diff}, step=i_iter)

    #     if i_iter >= args.num_steps_stop - 1:
    #         print('save model ...')
    #         if args.use_wandb:
    #             torch.save(model.state_dict(), osp.join(wandb.run.dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
    #         else:
    #             torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
    #         break

    #     if i_iter % args.save_pred_every == 0 and i_iter != 0:
    #         print('taking snapshot ...')
    #         if args.use_wandb:
    #             torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
    #         else:
    #             torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()
