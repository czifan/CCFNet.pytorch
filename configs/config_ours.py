from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np

class Config(object):
    def __init__(self):
        super().__init__()
        # gpu | cpu settings
        self.gpus = [0, ]
        self.device = torch.device('cuda:{}'.format(self.gpus[0])
                                   if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.num_workers = 3 * len(self.gpus)
        else:
            self.num_workers = 0

        # processing data settings
        self.multi_processing = True
        self.is_merge_valid = True
        self.train_data_dir = 'preliminary_dataset/lumbar_train201/'
        self.train_json_file = 'preliminary_dataset/lumbar_train201_annotation.json'
        self.valid_data_dir = 'preliminary_dataset/lumbar_train201/'
        self.valid_json_file = 'preliminary_dataset/lumbar_train201_annotation.json'
        self.testB_data_dir = 'preliminary_dataset/lumbar_testB50/'
        self.testB_json_file = 'preliminary_dataset/testB50_series_map.json'
        self.testAB_data_dir = 'preliminary_dataset/lumbar_testAB100/'
        self.testAB_json_file = 'preliminary_dataset/lumbar_testAB100_annotation.json'
        self.kfold_model_dir = 'models/kfold/'

        # network settings
        self.backbone = 'resnet18'
        self.pretrained = 'models/imagenet/resnet18-5c106cde.pth'
        #self.pretrained = 'models/imagenet/resnet50-19c8e357.pth'

        # dataset settings
        self.num_rep = 10  # 每一个epoch遍历数据的次数
        self.num_classes = 11  # 关键点的类别数（5个锥体+6个椎间盘）
        self.num_v_classes = 2
        self.v_weights = 1. / np.array([191, 814])
        self.v_numbers = [191, 814]
        self.num_d_classes = 4
        self.d_weights = 1. / np.array([574, 300, 285, 47])
        self.d_numbers = [574, 300, 285, 47]
        self.num_d_v5_classes = 2
        self.d_v5_weights = 1. / np.array([1142, 64])
        self.d_v5_numbers = [1142, 64]
        self.sagittal_size = (512, 512)
        self.predict_size = (32, 32)
        self.transverse_size = (80, 80)
        self.stride = 16
        self.sigma = 16

        # training settings
        self.batch_size = 32 * len(self.gpus)
        self.epochs = 60
        self.display = 200
        self.lr = 3e-4
        self.weight_decay = 5e-4
        self.gamma = 0.96
        self.flooding_d = 0.0008
        self.test_size = 0.25

        # evaluate
        self.test_flip = True
        self.heat_weight = 1.0
        self.offset_weight = 2.0
        self.class_weight = 0.001
        self.max_dist = 6
        self.epsilon = 1e-5
        self.top_k = 1
        self.vote = 'average'
        self.train_threshold = 0.6
        self.threshold = 0.3
        self.metric = 'macro f1'
        self.identifications = ['L1', 'L2', 'L3', 'L4', 'L5', 'T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']
        self.classes = ['v2', 'v2', 'v2', 'v2', 'v2', 'v1', 'v1', 'v1', 'v2', 'v3', 'v3']
        self.is_disc = [False, False, False, False, False, True, True, True, True, True, True]
        self.trans_matrix = None

        # augmentation settings
        self.sagittal_trans_settings = {
            'size': self.sagittal_size,
            'rotate_p': 0.8, 'max_angel': 45,
            'shift_p': 0.8, 'max_shift_ratios': [0.2, 0.3],
            'crop_p': 0.8, 'max_crop_ratios': [0.2, 0.3],
            'intensity_p': 0.8, 'max_intensity_ratio': 0.3,
            'bias_field_p': 0.8, 'order': 3, 'coefficients_range': [-0.5, 0.5],
            'noise_p': 0., 'noise_mean': [-5, 5], 'noise_std': [0, 2],
            'hflip_p': 0.5
        }
        self.transverse_trans_settings = {
            'size': self.transverse_size,
            'rotate_p': 0.8, 'max_angel': 25,
            'shift_p': 0.8, 'max_shift_ratios': [0.1, 0.1],
            'crop_p': 0.8, 'max_crop_ratios': [0.1, 0.1],
            'intensity_p': 0.8, 'max_intensity_ratio': 0.3,
            'bias_field_p': 0.8, 'order': 3, 'coefficients_range': [-0.5, 0.5],
            'noise_p': 0., 'noise_mean': [-5, 5], 'noise_std': [0, 2],
            'hflip_p': 0.5
        }

        # informations
        self.DICOM_TAG = {"studyUid": "0020|000d",
                          "seriesUid": "0020|000e",
                          "instanceUid": "0008|0018",
                          "pixelSpacing": "0028|0030",
                          "seriesDescription": "0008|103e",
                          "imagePosition": "0020|0032",
                          "imageOrientation": "0020|0037"}

        self.SPINAL_VERTEBRA_ID = {"L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4}
        self.SPINAL_VERTEBRA_DISEASE_ID = {"v1": 0, "v2": 1}
        self.SPINAL_DISC_ID = {"T12-L1": 0, "L1-L2": 1, "L2-L3": 2, "L3-L4": 3, "L4-L5": 4, "L5-S1": 5}
        self.SPINAL_DISC_DISEASE_ID = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4}
        self.PADDING_VALUE: int = -1

config = Config()
