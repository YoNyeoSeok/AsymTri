import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from model.deeplab import affine_par, conv3x3, BasicBlock, Bottleneck
# from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from model.deeplabv3 import DeepLabHead


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
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
        self.layer5 = self._make_pred_layer(1024, num_classes)
        self.layer6 = self._make_pred_layer(2048, num_classes)

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

    def _make_pred_layer(self, inplanes, num_classes):
        head = DeepLabHead(inplanes, num_classes)

        aspp, _, norm, _, _ = head._modules.values()

        convs_pool = aspp._modules['convs']
        convs, pool = convs_pool[:-1], convs_pool[-1]
        for conv in convs:
            conv._modules['1'].affine = affine_par
            for i in conv._modules['1'].parameters():
                i.requires_grad = False
        pool._modules['2'].affine = affine_par
        for i in pool._modules['2'].parameters():
            i.requires_grad = False

        proj = aspp._modules['project']
        proj._modules['1'].affine = affine_par
        for i in proj._modules['1'].parameters():
            i.requires_grad = False

        norm.affine = affine_par
        for i in norm.parameters():
            i.requires_grad = False

        return head
    
    def train(self):
        super(ResNetMulti, self).train()
        for layer in [self.layer5, self.layer6]:
            layer._modules['0']._modules['convs']._modules['4']._modules['2'].eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x1 = self.layer5(x)

        x2 = self.layer4(x)
        x2 = self.layer6(x2)

        return x1, x2

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
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def DeeplabV3Multi(num_classes=21):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
