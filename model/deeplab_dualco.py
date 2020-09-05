import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from model.deeplab import affine_par, conv3x3, BasicBlock, Bottleneck
from model.deeplab_multi import Classifier_Module

from matplotlib import pyplot as plt


class ResNetDualCo(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetDualCo, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(
            Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(
            Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer7 = self._make_pred_layer(
            Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer8 = self._make_pred_layer(
            Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.layer5(x)
        x2 = self.layer6(x)
        x3 = self.layer7(x)
        x4 = self.layer8(x)

        return x1, x2, x3, x4

    def weight_diff(self, orth=[], norm=1):
        """
            different feature criteria:
            select : $$ \sum |W| \odot |V| = 0 $$
            orthogonal : $$ W \odot V = 0 $$
            loss type: L1 or L2
        """
        diff = 0
        for m1, m2 in zip(self.layer5.conv2d_list, self.layer6.conv2d_list):
            for (n1, w1), (n2, w2) in zip(m1.named_parameters(), m2.named_parameters()):
                assert n1 == n2, "classifier not match"
                if 'weight' in n1:
                    vec = w1 * w2
                    diff += (vec.sum(orth) if orth else vec).norm(norm).pow(norm)

        return diff

    def weight_viz(self, suptitle, save_path):
        save_name_list = []
        for cn, (m1, m2) in enumerate(zip(self.layer5.conv2d_list, self.layer6.conv2d_list)):
            for (n1, w1), (n2, w2) in zip(m1.named_parameters(), m2.named_parameters()):
                assert n1 == n2, "classifier not match"
                if 'weight' in n1:
                    B, C, K1, K2 = w1.shape
                    w1w2 = torch.cat([w1.reshape(B, -1), w2.reshape(B, -1)], 1)
                    w1w2r = w1w2.abs().detach().cpu().numpy().max(1)

                    vec = w1 * w2
                    conflict = vec.abs()  # -1, C, K1, K2
                    c_conflict = vec.sum([1], keepdim=True).abs() # -1, 1, K1, K2
                    k_conflict = vec.sum([2, 3], keepdim=True).abs()  # -1, C, 1, 1
                    ck_conflict = vec.sum([1, 2, 3], keepdim=True).abs()  # -1, 1, 1, 1
                    conflicts = torch.cat([torch.cat([conflict, c_conflict], axis=1).reshape(-1, C+1, K1*K2),
                                        torch.cat([k_conflict, ck_conflict], axis=1).reshape(-1, C+1, 1),
                                        ], axis=2)   # -1, C+1, K1*K2+1

                    w1 = w1.detach().cpu().numpy()
                    w2 = w2.detach().cpu().numpy()
                    
                    cf = conflicts.detach().cpu().numpy()
                    cfr = conflicts.abs().detach().cpu().numpy().reshape(B, -1).max(1)

                    fig, axs = plt.subplots(3, 10, sharex=True, sharey=True, figsize=(K1*K2*10//3, K1*K2*3//3))
                    for k in range(10):
                        axs[0, k].imshow(w1[k].reshape(C, K1*K2)[-K1*K2+1:].T, 'RdGy', vmin=-w1w2r[k], vmax=w1w2r[k])
                        axs[0, k].set_title('layer5_{}{}'.format(n1, k))
                        axs[1, k].imshow(w2[k].reshape(C, K1*K2)[-K1*K2+1:].T, 'RdGy', vmin=-w1w2r[k], vmax=w1w2r[k])
                        axs[1, k].set_title('layer6_{}{}'.format(n2, k))
                        axs[2, k].imshow(cf[k][-K1*K2:].T, 'RdGy', vmin=-cfr[k], vmax=cfr[k])
                        axs[2, k].set_title('confilcts_{}{}'.format(n1, k))
                    plt.suptitle('{}_{}'.format(m1, suptitle))
                    plt.savefig('{}/conv{}_{}_sample.jpg'.format(save_path, cn, n1))
                    plt.close()
                    save_name_list.append('conv{}_{}_sample.jpg'.format(cn, n1))
        return save_name_list

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.layer5)
        b.append(self.layer6)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer7.parameters())
        b.append(self.layer8.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def DeeplabDualCo(num_classes=21):
    model = ResNetDualCo(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
