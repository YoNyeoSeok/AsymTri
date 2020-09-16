import argparse

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np


import torch.optim as optim

import torch.backends.cudnn as cudnn

import os
import os.path as osp
from deeplab.model import Res_Deeplab

from deeplab.datasets import GTA5DataSet


import timeit


start = timeit.default_timer()

IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32)  # BGR
IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32)  # BGR
BATCH_SIZE = 4
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/list/gta5/train.lst'
NUM_CLASSES = 19
IGNORE_LABEL = 255
INPUT_SIZE = '500,500'
TRAIN_SCALE = '0.5,1.5'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_STEPS = 100000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './gta_src_train/'
WEIGHT_DECAY = 0.0005
MODEL = 'DeeplabRes101'
LOG_FILE = 'log'
PIN_MEMORY = True
GPU = '0'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="The base network.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--train-scale", type=str, default=TRAIN_SCALE,
                        help="The scale for multi-scale training.")
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
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--pin-memory", type=bool, default=PIN_MEMORY,
                        help="Whether to pin memory in train & eval.")
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help="The name of log file.")
    parser.add_argument('--debug', help='True means logging debug info.',
                        default=False, action='store_true')
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for l in b:
        for m in l.modules():
            for p in m.parameters():
                if p.requires_grad:
                    yield p


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for p in b:
        for i in p:
            yield i


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    lscale, hscale = map(float, args.train_scale.split(','))
    train_scale = (lscale, hscale)

    cudnn.enabled = True

    # Create network.
    model = Res_Deeplab(num_classes=args.num_classes)

    new_params = model.state_dict().copy()
    if args.restore_from[:4] == 'http':
        saved_state_dict = torch.utils.model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    missing_keys = set(k for k in saved_state_dict.keys()
                       if k not in model.state_dict().keys())
    if missing_keys == set(('fc.weight', 'fc.bias')):
        for i in saved_state_dict:
            if i not in missing_keys:
                new_params[i] = saved_state_dict[i]
    elif missing_keys == set(('layer5')):
        pass
    else:
        raise NotImplementedError(
            "Can't guess possible restore model from url")
    model.load_state_dict(new_params)
    if args.is_training:
        model.train()
    else:
        model.eval()
    device = torch.device("cuda:" + str(args.gpu))
    model.to(device)

    cudnn.benchmark = True

    trainloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size, train_scale=train_scale,
                                              scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN, std=IMG_STD),
                                  batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=args.pin_memory)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate},
                           {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}],
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        pred = interp(model(images))
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        print('iter [{:8d}/{:8d}]: loss {}'.format(
            i_iter, args.num_steps, loss.data.cpu().numpy()))

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(
                args.snapshot_dir, 'GTA5_'+str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(
                args.snapshot_dir, 'GTA5_'+str(i_iter)+'.pth'))

    end = timeit.default_timer()
    print(end-start, 'seconds')


if __name__ == '__main__':
    main()
