import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from tqdm import tqdm
# from packaging import version
import time

import torch
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_diff import DeeplabDiff
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes'
DATA_LIST_PATH = './dataset/cityscapes_list/train.txt'
SAVE_PATH = './psLabel/cityscapes/train/'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500  # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'train'
PSTYPE = "and"
THTYPE = "class_90_100"

MODEL = 'DeeplabDiff'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def save_json(obj_f):
    torch.save(obj_f[0], obj_f[1])


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL, required=True,
                        help="Model Choice (DeeplabMulti/DeeplabDiff/DeeplabVGG/Oracle).")
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
    # parser.add_argument("--wandb-id", type=str, required=True)
    parser.add_argument("--pstype", type=str, default=PSTYPE,
                        help="pseudo-label type (1, 2, and, or) default: and")
    parser.add_argument("--thtype", type=str, default=THTYPE,
                        help="pseduo-label threshold type (class_\{02d\}_100, \{02d\}_100) default: class_90_100")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'DeeplabDiff':
        model = DeeplabDiff(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG
    else:
        assert False, "Not known model {}".format(args.model)

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    # for running different versions of pytorch
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k,
                        v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    ###
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                 batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(
        size=(1024, 2048), mode='bilinear', align_corners=True)
    ABC = 0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader), total=len(testloader)):
            image, _, name = batch
            # assert len(image) == 1, "Should have 1 batch size"
            if args.model == 'DeeplabDiff':
                output1, output2 = model(image.cuda(gpu0))
                output1 = interp(output1)[0]
                output2 = interp(output2)[0]
            elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
                output = model(image.cuda(gpu0))
                output = interp(output)[0]
                assert False, "Modified to eval two classifier"

            # outptu1 = (C x H x W)
            # output2 = (C x H x W)
            max_output1, argmax_output1 = output1.max(0)
            max_output2, argmax_output2 = output2.max(0)
            # max_output1 = np.asarray(np.max(output1, axis=2))
            # max_output2 = np.asarray(np.max(output2, axis=2))
            per_class_max_output1 = list(map(lambda c: max_output1.flatten()[
                                         argmax_output1.flatten() == c], range(19)))
            per_class_max_output2 = list(map(lambda c: max_output2.flatten()[
                                         argmax_output1.flatten() == c], range(19)))
            # print(list(map(lambda c: len(per_class_max_output1[c]), range(19))))

            threshold_idx_output1 = torch.linspace(
                0, len(max_output1.flatten()), 11, device=gpu0)[:-1].round().long()
            threshold_idx_output2 = torch.linspace(
                0, len(max_output2.flatten()), 11, device=gpu0)[:-1].round().long()
            per_class_threshold_idx_output1 = list(map(lambda c: torch.linspace(
                0, len(per_class_max_output1[c]), 11, device=gpu0)[:-1].round().long(), range(19)))
            per_class_threshold_idx_output2 = list(map(lambda c: torch.linspace(
                0, len(per_class_max_output2[c]), 11, device=gpu0)[:-1].round().long(), range(19)))
            # print(np.array(per_class_threshold_idx_output1))

            threshold_output1 = torch.cat([max_output1.flatten().sort(
            )[0], torch.tensor([1.], device=gpu0)])[threshold_idx_output1]
            threshold_output2 = torch.cat([max_output2.flatten().sort(
            )[0], torch.tensor([1.], device=gpu0)])[threshold_idx_output2]
            per_class_threshold_output1 = torch.stack(list(map(
                lambda c: torch.cat([per_class_max_output1[c].sort()[0], torch.tensor(
                    [1.], device=gpu0)])[per_class_threshold_idx_output1[c]],
                range(19))))
            per_class_threshold_output2 = torch.stack(list(map(
                lambda c: torch.cat([per_class_max_output2[c].sort()[0], torch.tensor(
                    [1.], device=gpu0)])[per_class_threshold_idx_output2[c]],
                range(19))))
            # print(np.array(per_class_threshold_output1))

            name = name[0].split('/')[-1]
            for idx, p in enumerate(range(0, 100, 10)):
                top_p = '{0:02d}_{1:d}'.format(p, 100)
                if args.thtype.strip('class_') != top_p:
                    continue

                # print('%s/%s_pred_%s \t %s' % (args.save, top_p, name, threshold_output1[idx]))
                if 'class' in args.thtype:
                    per_class_filter_output1 = argmax_output1.masked_fill(
                        mask=max_output1 < per_class_threshold_output1[argmax_output1, idx], value=IGNORE_LABEL)
                    per_class_filter_output2 = argmax_output2.masked_fill(
                        mask=max_output2 < per_class_threshold_output2[argmax_output2, idx], value=IGNORE_LABEL)
                    per_class_filter_output_and = per_class_filter_output1.masked_fill(
                        mask=per_class_filter_output1 != per_class_filter_output2, value=IGNORE_LABEL)
                    per_class_filter_output_or = per_class_filter_output_and.flatten().masked_scatter(
                        mask=per_class_filter_output1.flatten() == 255,
                        source=per_class_filter_output2.flatten()[per_class_filter_output1.flatten() == 255]).masked_scatter(
                            mask=per_class_filter_output2.flatten() == 255,
                            source=per_class_filter_output1.flatten()[per_class_filter_output2.flatten() == 255]).reshape(
                                *per_class_filter_output_and.shape)
                    if args.pstype == '1':
                        output_ = per_class_filter_output1
                    elif args.pstype == '2':
                        output_ = per_class_filter_output2
                    elif args.pstype == 'and':
                        output_ = per_class_filter_output_and
                    elif args.pstype == 'or':
                        output_ = per_class_filter_output_or
                elif 'class' not in args.thtype:
                    filter_output1 = argmax_output1.masked_fill(
                        mask=max_output1 < threshold_output1[idx], value=IGNORE_LABEL)
                    filter_output2 = argmax_output2.masked_fill(
                        mask=max_output2 < threshold_output2[idx], value=IGNORE_LABEL)
                    filter_output_and = filter_output1.masked_fill(
                        mask=filter_output1 != filter_output2, value=IGNORE_LABEL)
                    filter_output_or = filter_output_and.flatten().masked_scatter(
                        mask=filter_output1.flatten() == 255,
                        source=filter_output2.flatten()[filter_output1.flatten() == 255]).masked_scatter(
                            mask=filter_output2.flatten() == 255,
                            source=filter_output1.flatten()[filter_output2.flatten() == 255]).reshape(
                                *filter_output_and.shape)
                    if args.pstype == '1':
                        output_ = filter_output1
                    elif args.pstype == '2':
                        output_ = filter_output2
                    elif args.pstype == 'and':
                        output_ = filter_output_and
                    elif args.pstype == 'or':
                        output_ = filter_output_or

                prefix = os.path.join(args.save, '_'.join([args.thtype, args.pstype]), name.split(
                    '_')[0], '_'.join(name.split('_')[:-1])) + '_psLabel'
                if not os.path.exists(os.path.dirname(prefix + '.png')):
                    os.makedirs(os.path.dirname(prefix + '.png'))

                output = torchvision.transforms.functional.to_pil_image(
                    output_.int().cpu())
                output_col = output.copy().convert('P')
                output_col.putpalette(palette)

                output.save(prefix + '_trainIds.png')
                output_col.save(prefix + '_color.png')
                break


if __name__ == '__main__':
    main()
