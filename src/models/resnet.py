"""Code adapted from https://github.com/RaptorMai/online-continual-learning/models/resnet.py
"""
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import logging as lg
import random as r
import matplotlib.pyplot as plt
from torch.nn.functional import relu, avg_pool2d
from scipy import optimize

from src.utils.utils import make_orthogonal

if torch.cuda.is_available():
    dev = "cuda:0"
elif torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"
device = torch.device(dev)

EPSILON = 1e-10

bn = True


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class UpConv(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(UpConv, self).__init__()
        self.operation = nn.Sequential(
            conv3x3(channel_in, channel_in, stride=2),
            conv1x1(channel_in, channel_in, stride=1),
            nn.BatchNorm2d(channel_in),
            nn.ReLU(),
            conv3x3(channel_in, channel_in, stride=1),
            conv1x1(channel_in, channel_out, stride=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.operation(x)

class Aggregator(nn.Module):
    def __init__(self, dim_in, number_stage):
        super(Aggregator, self).__init__()
        self.number_stage = number_stage
        self.proj_head = nn.Sequential(
            nn.Linear(number_stage * dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, number_stage)
        )

    def forward(self, feas, logits):
        aggregated_logits = []

        feature = torch.cat(feas, dim=1)
        logit = self.proj_head(feature)
        weights = F.softmax(logit, dim=1)
        weighted_logit = 0.
        for i in range(self.number_stage):
            weighted_logit += logits[i] * weights[:, i].unsqueeze(-1)
        aggregated_logits = weighted_logit
        return aggregated_logits

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if bn else nn.Identity()
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes) if bn else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if bn else nn.Identity()
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class cosLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(cosLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.scale = 0.09

    def forward(self, x):
        x_norm_ = torch.norm(x, p=2, dim=1)
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.000001)

        L_norm = torch.norm(self.L.weight, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized, weight_normalized.transpose(0, 1))
        scores = cos_dist / self.scale
        return scores


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, dim_in=512, instance_norm=False, input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(input_channels, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1) if bn else nn.Identity()
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.bn1 = nn.InstanceNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(dim_in, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        out = self.features(x)
        out = self.linear(out)
        return out

    def forward(self, x):
        out = self.features(x)
        return out

    def full(self, x):
        out = self.features(x)
        logits = self.linear(out)
        return out, logits

class ResNetPCR(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, dim_in=512, instance_norm=False, input_channels=3):
        super(ResNetPCR, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(input_channels, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1) if bn else nn.Identity()
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.bn1 = nn.InstanceNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.pcrLinear = cosLinear(dim_in, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = self.conv1(x)
        out = relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        out = self.features(x)
        out = self.pcrLinear(out)
        return out

    def forward(self, x):
        out = self.features(x)
        return out

    def full(self, x):
        out = self.features(x)
        logits = self.pcrLinear(out)
        return logits, out

def Reduced_ResNet18(nclasses, nf=20, bias=True, input_channels=3, instance_norm=False):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(
        BasicBlock, [2, 2, 2, 2],
        nclasses,
        nf,
        bias,
        input_channels=input_channels,
        instance_norm=instance_norm
    )


def ResNet18(nclasses, dim_in=512, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, dim_in=dim_in)


def ResNet18PCR(nclasses, dim_in=512, nf=64, bias=True):
    return ResNetPCR(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, dim_in=dim_in)

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''


def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)


def ResNet50(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)


def ResNet101(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)


def ResNet152(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)


class OCMResnet(nn.Module):
    """backbone + projection head"""

    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ResNet18(100, nf=nf, dim_in=dim_in)
        self.head_representation = nn.Sequential(
            nn.Linear(dim_in, proj_dim)
        )
        self.head_classification = nn.Sequential(
            nn.Linear(dim_in, n_classes)
        )

    def forward(self, x, is_simclr=False):
        features = self.encoder(x)
        if is_simclr:
            out = self.head_representation(features)
            return features, out
        else:
            return features

    def full(self, x):
        features = self.encoder(x)
        out = self.head_classification(features)
        return features, out

    def logits(self, x):
        features = self.encoder(x)
        out = self.head_classification(features)
        return out


class GSAResnet(nn.Module):
    """backbone + projection head"""

    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ResNet18(100, nf=nf, dim_in=dim_in)
        self.head_representation = nn.Sequential(
            nn.Linear(dim_in, proj_dim)
        )
        self.head_classification = nn.Sequential(
            nn.Linear(dim_in, n_classes)
        )

    def forward(self, x, is_simclr=False):

        features = self.encoder(x)

        if is_simclr:
            out = self.head_representation(features)
            return features, out
        else:
            out = self.head_classification(features)
            return features

    def logits(self, x):
        features = self.encoder(x)
        out = self.head_classification(features)
        return out

    def full(self, x):
        features = self.encoder(x)
        out = self.head_classification(features)
        return features, out


## extra models for ImageNet experiments

class ImageNet_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, dim_in=512, instance_norm=False, input_channels=3):
        super(ImageNet_ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = nn.Conv2d(input_channels, nf * 1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(nf * 1) if bn else nn.Identity()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.bn1 = nn.InstanceNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(dim_in, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        out = self.features(x)
        out = self.linear(out)
        return out

    def forward(self, x):
        out = self.features(x)
        return out

    def full(self, x):
        out = self.features(x)
        logits = self.linear(out)
        return out, logits


def ImageNet_ResNet18(nclasses, dim_in=512, nf=64, bias=True):
    return ImageNet_ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, dim_in=dim_in)

class ImageNet_ResNetPCR(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, dim_in=512, instance_norm=False, input_channels=3):
        super(ImageNet_ResNetPCR, self).__init__()
        self.in_planes = nf
        self.conv1 = nn.Conv2d(input_channels, nf * 1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(nf * 1) if bn else nn.Identity()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.bn1 = nn.InstanceNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.pcrLinear = cosLinear(dim_in, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = self.conv1(x)
        out = relu(self.bn1(out))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        out = self.features(x)
        out = self.pcrLinear(out)
        return out

    def forward(self, x):
        out = self.features(x)
        return out

    def full(self, x):
        out = self.features(x)
        logits = self.pcrLinear(out)
        return logits, out

def ImageNet_ResNet18PCR(nclasses, dim_in=512, nf=64, bias=True):
    return ImageNet_ResNetPCR(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, dim_in=dim_in)

class ImageNet_OCMResnet(nn.Module):
    """backbone + projection head"""

    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ImageNet_ResNet18(n_classes, nf=nf, dim_in=dim_in)
        self.head_representation = nn.Sequential(
            nn.Linear(dim_in, proj_dim)
        )
        self.head_classification = nn.Sequential(
            nn.Linear(dim_in, n_classes)
        )

    def forward(self, x, is_simclr=False):
        features = self.encoder(x)
        if is_simclr:
            out = self.head_representation(features)
            return features, out
        else:
            return features

    def full(self, x):
        features = self.encoder(x)
        out = self.head_classification(features)
        return features, out

    def logits(self, x):
        features = self.encoder(x)
        out = self.head_classification(features)
        return out


class ImageNet_GSAResnet(nn.Module):
    """backbone + projection head"""

    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ImageNet_ResNet18(n_classes, nf=nf, dim_in=dim_in)
        self.head_representation = nn.Sequential(
            nn.Linear(dim_in, proj_dim)
        )
        self.head_classification = nn.Sequential(
            nn.Linear(dim_in, n_classes)
        )

    def forward(self, x, is_simclr=False):

        features = self.encoder(x)

        if is_simclr:
            out = self.head_representation(features)
            return features, out
        else:
            out = self.head_classification(features)
            return features

    def logits(self, x):
        features = self.encoder(x)
        out = self.head_classification(features)
        return out

    def full(self, x):
        features = self.encoder(x)
        out = self.head_classification(features)
        return features, out
