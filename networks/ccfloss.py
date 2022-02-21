from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CBCrossEntropyLoss(nn.Module):
    def __init__(self, samples_per_class, device):
        super().__init__()
        self.samples_per_class = np.sum(np.stack([value for value in samples_per_class.values()], axis=0), axis=0)
        if not isinstance(self.samples_per_class, np.ndarray):
            self.samples_per_class = np.array(self.samples_per_class)
        self.betas_per_class = (self.samples_per_class - 1.) / self.samples_per_class
        self.ens_per_class = (1. - np.power(self.betas_per_class, self.samples_per_class)) / (1. - self.betas_per_class)
        self.weights_per_class = 1. / (self.ens_per_class + 1e-5)
        self.base_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_per_class).float().to(device))

    def forward(self, predictions, targets):
        return self.base_criterion(predictions, targets)
    
class CBBCEWithLogitsLoss(nn.Module):
    def __init__(self, samples_per_class, device):
        super().__init__()
        self.samples_per_class = np.sum(np.stack([value for value in samples_per_class.values()], axis=0), axis=0)
        if not isinstance(self.samples_per_class, np.ndarray):
            self.samples_per_class = np.array(self.samples_per_class)
        self.betas_per_class = (self.samples_per_class - 1.) / self.samples_per_class
        self.ens_per_class = (1. - np.power(self.betas_per_class, self.samples_per_class)) / (1. - self.betas_per_class)
        self.weights_per_class = 1. / (self.ens_per_class + 1e-5)
        self.pos_weight = self.weights_per_class[1] / (self.weights_per_class[0] + 1e-5)
        self.base_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]).float().to(device))
        
    def forward(self, predictions, targets):
        return self.base_criterion(predictions, targets)

class CCFHeatLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, predictions, targets, masks):
        B, C, _, _ = predictions.shape
        predictions = predictions * masks
        targets = targets * masks
        predictions = predictions.reshape((B, C, -1)).split(1, 1)
        targets = targets.reshape((B, C, -1)).split(1, 1)
        loss = 0
        for c in range(C):
            prediction = predictions[c].squeeze()
            target = targets[c].squeeze()
            loss += self.criterion(prediction, target)

        return loss / C

class CCFOffsetLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.SmoothL1Loss()
        self.threshold = config.train_threshold

    def forward(self, predictions, targets, masks):
        peaks = (masks.max(3))[0].max(2)[0]
        peaks = peaks.unsqueeze(2).unsqueeze(3)
        masks = masks - peaks
        masks[masks >= (-1 * self.threshold)] = 1
        masks[masks < (-1 * self.threshold)] = 0
        predictions = predictions * masks
        targets = targets * masks
        return self.criterion(predictions, targets)

class CCFLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heat_weight = config.heat_weight
        self.offset_weight = config.offset_weight
        self.class_weight = config.class_weight
        self.heat_criterion = CCFHeatLoss(config)
        self.offset_criterion = CFFOffsetLoss(config)
        self.threshold = config.train_threshold
        
        self.v_cls_criterion = CBBCEWithLogitsLoss(config.v_numbers, config.device)
        d_numbers = {}
        for key in config.d_numbers.keys():
            d_numbers[key] = np.asarray([config.d_numbers[key][0], sum(config.d_numbers[key][1:])]).astype(np.float)
        self.d_cls_criterion = CBBCEWithLogitsLoss(d_numbers, config.device)

    def forward(self, heat_predictions, offy_predictions, offx_predictions, clss_predictions,
                heat_targets, offy_targets, offx_targets, clss_targets, masks,
                v_loss_weight, d_loss_weight, d_v5_loss_weight=None, name='stage1'):
        heat_loss = self.heat_criterion(heat_predictions, heat_targets, masks)
        offy_loss = self.offset_criterion(offy_predictions, offy_targets, heat_targets)
        offx_loss = self.offset_criterion(offx_predictions, offx_targets, heat_targets)

        vx, vy, dx, dy = [], [], [], []
        temp_targets = heat_targets.detach().view(heat_targets.shape[0], heat_targets.shape[1], -1)
        for b in range(temp_targets.shape[0]):
            for c in range(temp_targets.shape[1]):
                max_index = torch.argmax(temp_targets[b, c]).item()
                basic_grid_y, basic_grid_x = int(max_index // heat_targets.shape[3]), int(
                    max_index % heat_targets.shape[3])
                for oy, ox in ((0, 0),):
                    grid_y = basic_grid_y + oy
                    grid_x = basic_grid_x + ox
                    if grid_y < 0 or grid_y >= clss_predictions.shape[2] or grid_x < 0 or grid_x >= \
                            clss_predictions.shape[3]:
                        continue
                    if c < 5:
                        if clss_targets[b, c, grid_y, grid_x].item() < 0: continue
                        vx.append(clss_predictions[b, c, grid_y, grid_x])
                        vy.append(clss_targets[b, c, grid_y, grid_x])
                    else:
                        if clss_targets[b, c, grid_y, grid_x].item() < 0: continue
                        dx.append(clss_predictions[b, c, grid_y, grid_x])
                        dy.append((clss_targets[b, c, grid_y, grid_x] >= 1).float())
        vx = torch.stack(vx, dim=0)  # (B * C)
        vy = torch.stack(vy, dim=0)  # (B * C,)
        dx = torch.stack(dx, dim=0)  # (B * C)
        dy = torch.stack(dy, dim=0)  # (B * C,)
        v_cls_loss = self.v_cls_criterion(vx.float(), vy.float())
        d_cls_loss = self.d_cls_criterion(dx.float(), dy.float())

        loss = heat_loss * self.heat_weight + offy_loss * self.offset_weight + offx_loss * self.offset_weight \
               + v_cls_loss * v_loss_weight + d_cls_loss * d_loss_weight
        info = {'loss': loss.item(),
                name + '_heat_loss': heat_loss.item(),
                name + '_offy_loss': offy_loss.item(),
                name + '_offx_loss': offx_loss.item(),
                name + '_v_cls_loss': v_cls_loss.item(),
                name + '_d_cls_loss': d_cls_loss.item()}
        return loss, info