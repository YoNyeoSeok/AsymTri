import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
# from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_tri import DeeplabTri
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabTri'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabTri/DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'DeeplabTri':
        model = DeeplabTri(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    # for running different versions of pytorch
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    ###
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(
        size=(1024, 2048), mode='bilinear', align_corners=True)

    with torch.no_grad():
        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd' % index)
            image, _, name = batch
            if args.model == 'DeeplabTri':
                output1, output2, output3 = model(image.cuda(gpu0))
                output1 = interp(output1).cpu().data[0].numpy()
                output2 = interp(output2).cpu().data[0].numpy()
            elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
                output = model(image.cuda(gpu0))
                output = interp(output).cpu().data[0].numpy()

            output1 = output1.transpose(1, 2, 0)
            output2 = output2.transpose(1, 2, 0)
            argmax_output1 = np.asarray(
                np.argmax(output1, axis=2), dtype=np.uint8)
            argmax_output2 = np.asarray(
                np.argmax(output2, axis=2), dtype=np.uint8)
            max_output1 = np.asarray(np.max(output1, axis=2))
            max_output2 = np.asarray(np.max(output2, axis=2))
            per_class_max_output1 = list(map(lambda c: max_output1.flatten()[
                                         argmax_output1.flatten() == c], range(19)))
            per_class_max_output2 = list(map(lambda c: max_output2.flatten()[
                                         argmax_output2.flatten() == c], range(19)))
            # print(list(map(lambda c: len(per_class_max_output1[c]), range(19))))

            threshold_idx_output1 = np.linspace(0, len(max_output1.flatten()), 11)[
                :-1].astype(np.int)
            threshold_idx_output2 = np.linspace(0, len(max_output2.flatten()), 11)[
                :-1].astype(np.int)
            per_class_threshold_idx_output1 = list(map(lambda c: np.linspace(
                0, len(per_class_max_output1[c]), 11)[:-1].astype(np.int), range(19)))
            per_class_threshold_idx_output2 = list(map(lambda c: np.linspace(
                0, len(per_class_max_output2[c]), 11)[:-1].astype(np.int), range(19)))
            # print(np.array(per_class_threshold_idx_output1))

            threshold_output1 = np.sort(max_output1.flatten())[
                threshold_idx_output1]
            threshold_output2 = np.sort(max_output2.flatten())[
                threshold_idx_output2]
            per_class_threshold_output1 = list(map(lambda c: [None]*10 if per_class_max_output1[c].size == 0 else np.sort(
                per_class_max_output1[c])[per_class_threshold_idx_output1[c]], np.arange(19)))
            per_class_threshold_output2 = list(map(lambda c: [None]*10 if per_class_max_output2[c].size == 0 else np.sort(
                per_class_max_output2[c])[per_class_threshold_idx_output2[c]], np.arange(19)))
            # print(np.array(per_class_threshold_output1))

            name = name[0].split('/')[-1]
            for idx, p in enumerate(range(0, 100, 10)):
                top_p = '{0:02d}_{1:d}'.format(p, 100)
                and_filter = np.vectorize(
                    lambda o1, o2: o1 if o1 == o2 else IGNORE_LABEL)
                or_filter = np.vectorize(lambda o1, o2: o1 if o1 == o2 else (
                    o2 if o1 == IGNORE_LABEL else (o1 if o2 == IGNORE_LABEL else IGNORE_LABEL)))

                top_p_filter = np.vectorize(
                    lambda am, m, t: am if m >= t else IGNORE_LABEL)
                filter_output1 = top_p_filter(
                    argmax_output1, max_output1, threshold_output1[idx]).astype(np.uint8)
                filter_output2 = top_p_filter(
                    argmax_output2, max_output2, threshold_output2[idx]).astype(np.uint8)
                filter_output_and = and_filter(
                    filter_output1, filter_output2).astype(np.uint8)
                filter_output_or = or_filter(
                    filter_output1, filter_output2).astype(np.uint8)

                t1 = np.array(per_class_threshold_output1)[:, idx]
                t2 = np.array(per_class_threshold_output2)[:, idx]
                per_class_top_p_filter1 = np.vectorize(lambda am, m: am if (
                    t1[am] is not None) and (m >= t1[am]) else IGNORE_LABEL)
                per_class_top_p_filter2 = np.vectorize(lambda am, m: am if (
                    t2[am] is not None) and (m >= t2[am]) else IGNORE_LABEL)
                per_class_filter_output1 = per_class_top_p_filter1(
                    argmax_output1, max_output1).astype(np.uint8)
                per_class_filter_output2 = per_class_top_p_filter2(
                    argmax_output2, max_output2).astype(np.uint8)
                per_class_filter_output_and = and_filter(
                    per_class_filter_output1, per_class_filter_output2).astype(np.uint8)
                per_class_filter_output_or = or_filter(
                    per_class_filter_output1, per_class_filter_output2).astype(np.uint8)
                # output3 = np.vectorize(lambda s, o: o if s else IGNORE_LABEL)(output1==output2, output1).astype(np.uint8)

                # print('%s/%s_pred_%s \t %s' % (args.save, top_p, name, threshold_output1[idx]))
                for pred_, output_ in zip(['1', '2', 'and', 'or'], [filter_output1, filter_output2, filter_output_and, filter_output_or]):
                    output_col = colorize_mask(output_)
                    output = Image.fromarray(output_)

                    if not os.path.exists('%s/%s_pred_%s' % (args.save, top_p, pred_)):
                        os.makedirs('%s/%s_pred_%s' %
                                    (args.save, top_p, pred_))
                    output.save('%s/%s_pred_%s/%s' %
                                (args.save, top_p, pred_, name))
                    output_col.save('%s/%s_pred_%s/%s_color.png' %
                                    (args.save, top_p, pred_, name.split('.')[0]))

                # print('%s/class_%s_pred_%s \t %s' % (args.save, top_p, name, threshold_output1[idx]))
                for pred_, output_ in zip(['1', '2', 'and', 'or'], [per_class_filter_output1, per_class_filter_output2, per_class_filter_output_and, per_class_filter_output_or]):
                    output_col = colorize_mask(output_)
                    output = Image.fromarray(output_)

                    if not os.path.exists('%s/class_%s_pred_%s' % (args.save, top_p, pred_)):
                        os.makedirs('%s/class_%s_pred_%s' %
                                    (args.save, top_p, pred_))
                    output.save('%s/class_%s_pred_%s/%s' %
                                (args.save, top_p, pred_, name))
                    output_col.save('%s/class_%s_pred_%s/%s_color.png' %
                                    (args.save, top_p, pred_, name.split('.')[0]))


if __name__ == '__main__':
    main()
