from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1,
                                                      stride=stride, padding=0, bias=False),
                                            nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM))
        else:
            self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DetnetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(DetnetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class Block1(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dilate1 = nn.Conv2d(ch, ch, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(ch, ch, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(ch, ch, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(ch, ch, kernel_size=1, dilation=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.conv1x1(self.dilate2(x)))
        dilate3_out = self.relu(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = self.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class Block2(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.poo1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.poo2 = nn.MaxPool2d(kernel_size=[4, 4], stride=4)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.poo3 = nn.MaxPool2d(kernel_size=[8, 8], stride=8)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        self.layer1 = self.up1(self.conv(self.poo1(x)))
        self.layer2 = self.up2(self.conv(self.poo2(x)))
        self.layer3 = self.up3(self.conv(self.poo3(x)))
        out = torch.cat([self.layer1, self.layer2, self.layer3, x], dim=1)
        return out

class HeadBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.feat = nn.Sequential(Block1(in_channels),
                                  Block2(in_channels),
                                  Bottleneck(in_channels + 3, in_channels // Bottleneck.expansion))
        self.out = nn.Conv2d(in_channels=in_channels // Bottleneck.expansion * Bottleneck.expansion,
                             out_channels=out_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.feat(x)
        x = self.out(x)

        heatmap = x[:, :self.num_classes]
        offymap = x[:, self.num_classes:self.num_classes * 2]
        offxmap = x[:, self.num_classes * 2:self.num_classes * 3]
        clssmap = x[:, self.num_classes * 3:]

        return heatmap, offymap, offxmap, clssmap