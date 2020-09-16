import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision.transforms as transforms
import torchvision
import cv2
from torch.utils import data
import sys
from PIL import Image

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


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * \
                int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s.jpg" % name)
            label_file = osp.join(self.root, "gt/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(
            img_pad[h_off: h_off+self.crop_h, w_off: w_off+self.crop_w], np.float32)
        label = np.asarray(
            label_pad[h_off: h_off+self.crop_h, w_off: w_off+self.crop_w], np.float32)
        # image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, pseudo_root=None, max_iters=None, crop_size=(500, 500), train_scale=(0.5, 1.5), mean=(128, 128, 128), std=(1, 1, 1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters == None:
            self.img_ids = self.img_ids * \
                int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * \
                int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + \
            random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        # OpenCV read image as BGR, not RGB
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = np.array(Image.open(datafiles["label"]))
        #
        sys.path.insert(0, 'dataset/helpers')
        from labels import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # valid_labels = sorted(set(id_2_label.ravel()))
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image/255.0  # scale to [0,1]
        image -= self.mean  # BGR
        image = image/self.std  # np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(
            img_pad[h_off: h_off+self.crop_h, w_off: w_off+self.crop_w], np.float32)
        label = np.asarray(
            label_pad[h_off: h_off+self.crop_h, w_off: w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), img_name


if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
