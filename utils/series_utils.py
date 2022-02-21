from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from collections import Counter
from typing import List
from utils.dicom_utils import DICOM
from utils.utils import lazy_property
from PIL import Image


class Series(list):
    def __init__(self, dicom_list: List[DICOM]):
        planes = [dicom.plane for dicom in dicom_list]
        plane_counter = Counter(planes)
        self.plane = plane_counter.most_common(1)[0][0]

        if self.plane == 'transverse':
            dim = 2
        elif self.plane == 'sagittal':
            dim = 0
        elif self.plane == 'transverse':
            dim = 1
        else:
            dim = None

        dicom_list = [dicom for dicom in dicom_list if dicom.plane == self.plane]
        if dim is not None:
            dicom_list = sorted(dicom_list, key=lambda x: x.image_position[dim], reverse=True)
        if self.plane == 'sagittal':
            images = [np.asarray(dicom.image) for dicom in dicom_list if dicom.image is not None]
            if len(images) > 0:
                images = [images[0]] + images + [images[-1]]
                for idx in range(len(dicom_list)):
                    try:
                        dicom_list[idx].image = Image.fromarray(
                            np.stack(images[idx + 1 - 1: idx + 1 + 2], axis=2).astype(np.uint8))
                    except:
                        dicom_list[idx].image = Image.fromarray(
                            np.stack([images[idx] for _ in range(3)], axis=2).astype(np.uint8))
        super().__init__(dicom_list)
        self.instance_uids = {d.instance_uid: i for i, d in enumerate(self)}
        self.instance_idxs = {i: d.instance_uid for i, d in enumerate(self)}
        self.middle_frame_uid = None

    def __getitem__(self, item) -> DICOM:
        if isinstance(item, str):
            item = self.instance_uids[item]
        return super().__getitem__(item)

    @lazy_property
    def t_type(self):
        t_type_counter = Counter([d.t_type for d in self])
        return t_type_counter.most_common(1)[0][0]

    @lazy_property
    def t_info(self):
        t_info_counter = Counter([d.t_info for d in self])
        return t_info_counter.most_common(1)[0][0]

    @lazy_property
    def mean(self):
        output = 0
        i = 0
        for dicom in self:
            mean = dicom.mean
            if mean is None:
                continue
            output = i / (i + 1) * output + mean / (i + 1)
            i += 1
        return output

    @property
    def middle_frame(self) -> DICOM:
        if self.middle_frame_uid is not None:
            return self[self.middle_frame_uid]
        else:
            return self[len(self) // 2]

    def set_middle_frame(self, instance_uid):
        self.middle_frame_uid = instance_uid

    @property
    def image_positions(self):
        positions = []
        for dicom in self:
            positions.append(dicom.image_position)
        return torch.stack(positions, dim=0)

    @property
    def unit_normal_vectors(self):
        vectors = []
        for dicom in self:
            vectors.append(dicom.unit_normal_vector)
        return torch.stack(vectors, dim=0)

    @lazy_property
    def series_uid(self):
        study_uid_counter = Counter([d.series_uid for d in self])
        return study_uid_counter.most_common(1)[0][0]

    @lazy_property
    def study_uid(self):
        study_uid_counter = Counter([d.study_uid for d in self])
        return study_uid_counter.most_common(1)[0][0]

    def point_distance(self, coord: torch.Tensor):
        return torch.stack([dicom.point_distance(coord) for dicom in self], dim=1).squeeze()

    def k_nearest(self, coord: torch.Tensor, k, max_dist) -> List[List[DICOM]]:
        distance = self.point_distance(coord)
        indices = torch.argsort(distance, dim=1)
        if len(indices) == 1:
            return [[self[i] if distance[i] < max_dist else None for i in indices[:k]]]
        else:
            return [[self[i] if row_d[i] < max_dist else None for i in row[:k]]
                    for row, row_d in zip(indices, distance)]
