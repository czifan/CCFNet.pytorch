from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from typing import Dict, Any, Tuple
from utils.study_utils import Study
from torch.utils.data import Dataset
from utils.dicom_utils import DICOM
from utils.cl_utils import cl_encoder
from glob import glob
import os
import numpy as np
from itertools import chain
import heapq

def prim(graph):
    n = len(graph)
    v = 0
    s = {v}
    edges = []
    res = []
    record = [v]
    for _ in range(n - 1):
        for u, w in graph[v].items():
            heapq.heappush(edges, (w, v, u))
        while edges:
            w, p, q = heapq.heappop(edges)
            if q not in s:
                s.add(q)
                record.append(q)
                res.append(((p, q), w))
                v = q
                break
    return res, record

def check_annotation(v_annotation, d_annotation, d_v5_annotation):
    def _distance(a, b):
        return (((a - b) ** 2).sum()) ** 0.5
    v_annotation = v_annotation.detach().cpu().numpy()
    d_annotation = d_annotation.detach().cpu().numpy()
    d_v5_annotation = d_v5_annotation.detach().cpu().numpy()
    temp = list(chain.from_iterable(zip(d_annotation[:-1, :2], v_annotation[:, :2]))) + [d_annotation[-1, :2]]
    graph = [[] for _ in range(len(temp))]
    for i in range(len(temp)):
        graph[i] = {}
        for j in range(len(temp)):
            if j == i: continue
            graph[i][j] = _distance(temp[i], temp[j])
    _, path = prim(graph)
    temp = [temp[i] for i in path]
    for i in range(len(temp)):
        if i % 2 == 0: # d
            d_annotation[i//2, :2] = temp[i]
            d_v5_annotation[i//2, :2] = temp[i]
        else:
            v_annotation[i//2, :2] = temp[i]
    v_annotation = torch.from_numpy(v_annotation)
    d_annotation = torch.from_numpy(d_annotation)
    d_v5_annotation = torch.from_numpy(d_v5_annotation)
    return v_annotation, d_annotation, d_v5_annotation

class SpineDataSet(Dataset):
    def __init__(self,
                 config,
                 studies: Dict[Any, Study],
                 annotations: Dict[Any, Tuple[torch.Tensor, torch.Tensor]]):
        super().__init__()
        self.config = config
        self.num_rep = config.num_rep
        self.studies = studies
        self.annotations = []
        for k, annotation in annotations.items():
            study_uid, series_uid, instance_uid = k
            if study_uid not in self.studies:
                continue
            study = self.studies[study_uid]
            if series_uid in study and instance_uid in study[series_uid].instance_uids:
                self.annotations.append((k, annotation))

    def __getitem__(self, item):
        item = item % len(self.annotations)
        key, (v_annotation, d_annotation, d_v5_annotation) = self.annotations[item]
        #v_annotation, d_annotation, d_v5_annotation = check_annotation(v_annotation, d_annotation, d_v5_annotation)
        return self.studies[key[0]], key, v_annotation, d_annotation, d_v5_annotation

    def collate_fn(self, data) -> (Tuple[torch.Tensor], Tuple[None]):
        sagittal_images = []
        transverse_images = []
        transverse_masks = []
        hflip_flags = []
        heatmaps, offymaps, offxmaps, maskmaps, clssmaps = [], [], [], [], []
        for study, key, v_anno, d_anno, d_v5_anno in data:
            instance_uid = study[key[1]].instance_idxs[study[key[1]].instance_uids[key[2]] + np.random.randint(-1, 2)]
            dicom: DICOM = study[key[1]][instance_uid]
#             random_p = np.random.uniform(0, 1)
#             if random_p < 1.0:
#                 instance_uid = study[key[1]].instance_idxs[study[key[1]].instance_uids[key[2]] + np.random.randint(-1, 2)]
#                 dicom: DICOM = study[key[1]][instance_uid]
#             elif random_p < 0.8:
#                 dicom: DICOM = study.t2_sagittal_middle_frame
#             else:
#                 dicom: DICOM = study.t1_sagittal_middle_frame
            pixel_coord = torch.cat([v_anno[:, :2], d_anno[:, :2]], dim=0)  # (11, 2)
            pixel_noise = torch.zeros_like(pixel_coord).float().uniform_(
                -self.config.max_dist / 4 / dicom.pixel_spacing[1].item(),
                self.config.max_dist / 4 / dicom.pixel_spacing[0].item())
            pixel_coord = pixel_coord + pixel_noise
            pixel_class = torch.cat([v_anno[:, -1], d_anno[:, -1], d_v5_anno[:, -1]])  # (11+6,)
            sagittal_image, pixel_coord, details = dicom.transform(self.config.sagittal_trans_settings, pixel_coord,
                                                                   tensor=True, isaug=True)
            sagittal_images.append(sagittal_image)

#             transverse_image, transverse_mask = study.t2_transverse_k_nearest(self.config, True, d_anno[:, :2], 1)
#             transverse_image = transverse_image[:, 0, :]
            transverse_image = torch.zeros(6, 1, 96, 96)
            transverse_mask = torch.zeros(6)
            transverse_images.append(transverse_image)  # (6, 1, 96, 96)
            transverse_masks.append(transverse_mask)

            heatmap, offymap, offxmap, maskmap, clssmap = cl_encoder(self.config, pixel_coord, pixel_class, tensor=True)
            heatmaps.append(heatmap)
            offymaps.append(offymap)
            offxmaps.append(offxmap)
            maskmaps.append(maskmap)
            clssmaps.append(clssmap)

        sagittal_images = torch.stack(sagittal_images, dim=0)
        transverse_images = torch.stack(transverse_images, dim=0)  # (B, 6, 3, 80, 80)
        transverse_masks = torch.stack(transverse_masks, dim=0) # (B, 6)
        heatmaps = torch.stack(heatmaps, dim=0)
        offymaps = torch.stack(offymaps, dim=0)
        offxmaps = torch.stack(offxmaps, dim=0)
        maskmaps = torch.stack(maskmaps, dim=0)
        clssmaps = torch.stack(clssmaps, dim=0)

        return sagittal_images, heatmaps, offymaps, offxmaps, maskmaps, clssmaps, transverse_images, transverse_masks

    def __len__(self):
        return len(self.annotations) * self.num_rep
