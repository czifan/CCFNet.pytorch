from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import torch.nn as nn
import torch
from collections import OrderedDict

def init_weights(model, pretrained):
    if os.path.isfile(pretrained):
        print('=> init model weights as normal')
        pas = 0
        for name, m in model.named_modules():
            if 'd_model' in name or 'v_model' in name:
                pas += 1
                continue
            # if isinstance(m, nn.Conv2d):
            #     nn.init.normal_(m.weight, std=0.001)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        suc = 0
        model_state_dict = model.state_dict()
        loaded_state_dict = torch.load(pretrained)
        for key in loaded_state_dict.keys():
            new_key = key.replace('module.', '')
            if new_key in model_state_dict and loaded_state_dict[key].shape == model_state_dict[new_key].shape:
                model_state_dict[new_key] = loaded_state_dict[key]
                suc += 1
        model.load_state_dict(model_state_dict, strict=True)
        print('=> loaded pretrained model {}: {}/{} [{}]'.format(pretrained, suc, len(model_state_dict.keys()), pas))
    else:
        print('=> imagenet pretrained model dose not exist')
    return model

def get_ccfnet(config):
    if 'resnet' in config.backbone:
        from networks.ccfnet_resnet import get_ccfnet
        model = get_ccfnet(config)
    else:
        print('Undefined model: {}'.format(config.backbone))
        sys.exit()
    model = init_weights(model, config.pretrained)
    return model

def get_ccfnet_resnet(config):
    if 'resnet' in config.backbone:
        from networks.ccfnet_resnet import get_ccfnet_resnet
        model = get_ccfnet_resnet(config)
    else:
        print('Undefined model: {}'.format(config.backbone))
        sys.exit()
    model = init_weights(model, config.pretrained)
    return model