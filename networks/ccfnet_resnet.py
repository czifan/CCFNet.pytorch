from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from networks.parts import BasicBlock, Bottleneck, DetnetBottleneck, Block1, Block2, BN_MOMENTUM, HeadBlock

class CCFNet(nn.Module):
    def __init__(self, config, block, layers, channels):
        super().__init__()
        self.inplanes = 64
        self.num_classes = config.num_classes
        self.num_v_classes = config.num_v_classes
        self.num_d_classes = config.num_d_classes
        self.num_d_v5_classes = config.num_d_v5_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer5 = self._make_detnet_layer(channels, channels)
        self.layer_conv1x1 = nn.Conv2d(channels, 256, kernel_size=1, padding=0, stride=1)
        #out_channels = self.num_classes * 3 + (5 + 6 * self.num_d_classes + 6)
        out_channels = self.num_classes * 3 + (5 + 6)
        self.stage1_headnet = HeadBlock(in_channels=256, out_channels=out_channels, num_classes=self.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels, out_channels):
        layers = []
        layers.append(DetnetBottleneck(in_planes=in_channels, planes=out_channels, block_type='B'))
        layers.append(DetnetBottleneck(in_planes=out_channels, planes=out_channels, block_type='A'))
        layers.append(DetnetBottleneck(in_planes=out_channels, planes=out_channels, block_type='A'))
        return nn.Sequential(*layers)

    def _make_conv_bn_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        #x4 = self.layer5(x3)
        x3 = self.layer_conv1x1(x3)
        x4 = x3

        predictions = {}
        heatmap, offymap, offxmap, clssmap = self.stage1_headnet(x4)
        predictions['result'] = [heatmap, offymap, offxmap, clssmap]

        return predictions

class CCFNetResNet(CCFNet):
    def __init__(self, config, block, layers, channels):
        super().__init__(config, block, layers, channels)
        self.inplanes = 64
        self.num_classes = config.num_classes
        self.num_v_classes = config.num_v_classes
        self.num_d_classes = config.num_d_classes
        self.num_d_v5_classes = config.num_d_v5_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        out_channels = self.num_classes * 3 + (5 + 6)
        self.final_layer = nn.Conv2d(in_channels=512 if block == BasicBlock else 2048,
                             out_channels=out_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_layer(x)
        
        heatmap = x[:, :self.num_classes]
        offymap = x[:, self.num_classes:self.num_classes * 2]
        offxmap = x[:, self.num_classes * 2:self.num_classes * 3]
        clssmap = x[:, self.num_classes * 3:]

        predictions = {}
        predictions['result'] = [heatmap, offymap, offxmap, clssmap]

        return predictions

resnet_spec = {'resnet18': (BasicBlock, [2, 2, 2, 2], 256),
               'resnet50': (Bottleneck, [3, 4, 6, 3], 1024)}

def get_ccfnet(config):
    block_class, layers, channels = resnet_spec[config.backbone]
    model = CCFNet(config, block_class, layers, channels)
    return model

def get_ccfnet_resnet(config):
    block_class, layers, channels = resnet_spec[config.backbone]
    model = CCFNetResNet(config, block_class, layers, channels)
    return model
