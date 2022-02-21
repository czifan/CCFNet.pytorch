from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import Dict, Union
from utils.utils import lazy_property
from utils.dicom_utils import DICOM
from utils.series_utils import Series
from utils.utils import read_annotation
from torchvision.transforms import functional as tf


class Study(dict):
    def __init__(self, config, study_dir, pool=None):
        dicom_list = []
        if pool is not None:
            async_results = []
            for dicom_name in os.listdir(study_dir):
                dicom_path = os.path.join(study_dir, dicom_name)
                async_results.append(pool.apply_async(DICOM, (dicom_path, config.DICOM_TAG,)))

            for async_result in async_results:
                async_result.wait()
                dicom = async_result.get()
                dicom_list.append(dicom)
        else:
            for dicom_name in os.listdir(study_dir):
                dicom_path = os.path.join(study_dir, dicom_name)
                dicom = DICOM(dicom_path, config.DICOM_TAG)
                dicom_list.append(dicom)

        dicom_dict = {}
        for dicom in dicom_list:
            series_uid = dicom.series_uid
            if series_uid not in dicom_dict:
                dicom_dict[series_uid] = [dicom]
            else:
                dicom_dict[series_uid].append(dicom)

        super().__init__({k: Series(v) for k, v in dicom_dict.items()})

        self.t1_sagittal_uid = None
        self.t2_sagittal_uid = None
        self.t2_transverse_uid = None
        max_t1_sagittal_mean = 0
        max_t2_sagittal_mean = 0
        max_t2_transverse_mean = 0
        for series_uid, series in self.items():
            if series.plane == 'sagittal' and series.t_type == 'T2':
                t2_sagittal_mean = series.mean
                if t2_sagittal_mean > max_t2_sagittal_mean:
                    max_t2_sagittal_mean = t2_sagittal_mean
                    self.t2_sagittal_uid = series_uid
            if series.plane == 'transverse' and series.t_type == 'T2':
                t2_transverse_mean = series.mean
                if t2_transverse_mean > max_t2_transverse_mean:
                    max_t2_transverse_mean = t2_transverse_mean
                    self.t2_transverse_uid = series_uid
            if series.plane == 'sagittal' and series.t_type == 'T1':
                t1_sagittal_mean = series.mean
                if t1_sagittal_mean > max_t1_sagittal_mean:
                    max_t1_sagittal_mean = t1_sagittal_mean
                    self.t1_sagittal_uid = series_uid

        if self.t2_sagittal_uid is None:
            for series_uid, series in self.items():
                if series.plane == 'sagittal':
                    t2_sagittal_mean = series.mean
                    if t2_sagittal_mean > max_t2_sagittal_mean:
                        max_t2_sagittal_mean = t2_sagittal_mean
                        self.t2_sagittal_uid = series_uid

        if self.t2_transverse_uid is None:
            for series_uid, series in self.items():
                if series.plane == 'transverse':
                    t2_transverse_mean = series.mean
                    if t2_transverse_mean > max_t2_transverse_mean:
                        max_t2_transverse_mean = t2_transverse_mean
                        self.t2_transverse_uid = series_uid

        if self.t1_sagittal_uid is None:
            self.t1_sagittal_uid = self.t2_sagittal_uid

    @lazy_property
    def study_uid(self):
        study_uid_counter = Counter([s.study_uid for s in self.values()])
        return study_uid_counter.most_common(1)[0][0]

    @property
    def t2_sagittal(self) -> Union[None, Series]:
        if self.t2_sagittal_uid is None:
            return None
        else:
            return self[self.t2_sagittal_uid]

    @property
    def t1_sagittal(self) -> Union[None, Series]:
        if self.t1_sagittal_uid is None:
            return None
        else:
            return self[self.t1_sagittal_uid]

    @property
    def t2_transverse(self) -> Union[None, Series]:
        if self.t2_transverse_uid is None:
            return None
        else:
            return self[self.t2_transverse_uid]

    @property
    def t2_sagittal_middle_frame(self) -> Union[None, DICOM]:
        if self.t2_sagittal is None:
            return None
        else:
            return self.t2_sagittal.middle_frame

    @property
    def t1_sagittal_middle_frame(self) -> Union[None, DICOM]:
        if self.t1_sagittal is None:
            return None
        else:
            return self.t1_sagittal.middle_frame

    @property
    def t2_sagittal_middle_frame_off(self) -> Union[None, DICOM]:
        if self.t2_sagittal is None:
            return None
        else:
            return self.t2_sagittal.middle_frame_off

    def set_t2_sagittal_middle_frame(self, series_uid, instance_uid):
        assert series_uid in self
        self.t2_sagittal_uid = series_uid
        self.t2_sagittal.set_middle_frame(instance_uid)

    def _gen_t2_transverse_one_image_mask(self, config, point, dicom):
        if dicom is None:
            mask = True
            image = torch.zeros(1, *config.transverse_size)
        else:
            mask = False
            projection = dicom.projection(point)
            pixel_spacing = dicom.pixel_spacing
            image, projection = dicom.transform(config.transverse_trans_settings, projection, tensor=False, isaug=True)
            if len(projection.shape) == 2:
                projection = projection[0]
            transverse_size = (int(config.transverse_size[0] / pixel_spacing[0].item()),
                               int(config.transverse_size[1] / pixel_spacing[1].item()))
            image = tf.crop(image,
                            int(projection[1] - transverse_size[0] // 2),
                            int(projection[0] - transverse_size[1] // 2),
                            transverse_size[0],
                            transverse_size[1])
            image = tf.resize(image, config.transverse_size)
            image = tf.to_tensor(image)
        return image, mask

    def _gen_t2_transverse_batch_image_mask(self, config, point, series):
        results = list(
            map(self._gen_t2_transverse_one_image_mask, [config] * len(series), [point] * len(series), series))
        temp_images, temp_masks = [], []
        for image, mask in results:
            temp_images.append(image)
            temp_masks.append(mask)
        temp_images = torch.stack(temp_images, dim=0)
        return temp_images, temp_masks

    def t2_transverse_k_nearest(self, config, isaug, pixel_coord, k):
        if k <= 0 or self.t2_transverse is None:
            images = torch.zeros(pixel_coord.shape[0], k, 1, *config.transverse_size)
            masks = torch.zeros(*images.shape[:2], dtype=torch.bool)
            return images, masks
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord)
        dicoms = self.t2_transverse.k_nearest(human_coord, k, config.max_dist * 2)
        images = []
        masks = []
        for point, series in zip(human_coord, dicoms):
            temp_images = []
            temp_masks = []
            for dicom in series:
                if dicom is None:
                    temp_masks.append(False)
                    image = torch.zeros(1, *config.transverse_size)
                else:
                    temp_masks.append(True)
                    projection = dicom.projection(point)
                    pixel_spacing = dicom.pixel_spacing
                    image, projection = dicom.transverse_transform(
                        config.transverse_trans_settings, projection, tensor=True, isaug=isaug)
                temp_images.append(image)
            temp_images = torch.stack(temp_images, dim=0)
            images.append(temp_images)
            masks.append(temp_masks)
        images = torch.stack(images, dim=0)
        masks = torch.tensor(masks, dtype=torch.bool)
        return images, masks

def _construct_studies(config, data_dir, multiprocessing=False):
    studies: Dict[str, Study] = {}
    if multiprocessing:
        pool = Pool(cpu_count())
    else:
        pool = None

    for study_name in tqdm(os.listdir(data_dir), ascii=True):
        study_dir = os.path.join(data_dir, study_name)
        study = Study(config, study_dir, pool)
        studies[study.study_uid] = study

    if pool is not None:
        pool.close()
        pool.join()

    return studies

def _set_middle_frame(studies: Dict[str, Study], annotation):
    counter = {
        't2_sagittal_not_found': [],
        't2_sagittal_miss_match': [],
        't2_sagittal_middle_frame_miss_match': []
    }
    for k in annotation.keys():
        if k[0] in studies:
            study = studies[k[0]]
            if study.t2_sagittal is None:
                counter['t2_sagittal_not_found'].append(study.study_uid)
            elif study.t2_sagittal_uid != k[1]:
                counter['t2_sagittal_miss_match'].append(study.study_uid)
            else:
                t2_sagittal = study.t2_sagittal
                gt_z_index = t2_sagittal.instance_uids[k[2]]
                middle_frame = t2_sagittal.middle_frame
                z_index = t2_sagittal.instance_uids[middle_frame.instance_uid]
                if abs(gt_z_index - z_index) > 1:
                    counter['t2_sagittal_middle_frame_miss_match'].append(study.study_uid)
            study.set_t2_sagittal_middle_frame(k[1], k[2])
    return counter

def _testB_set_middle_frame(studies: Dict[str, Study], annotation):
    for anno_dict in annotation:
        if anno_dict['studyUid'] in studies:
            study = studies[anno_dict['studyUid']]
            study.t2_sagittal_uid = anno_dict['seriesUid']
    return studies

def construct_studies(config, data_dir, annotation_path=None, multi_processing=False):
    studies = _construct_studies(config, data_dir, multi_processing)

    if 'testB' in data_dir or 'round2test' in data_dir:
        annotation = json.load(open(annotation_path, 'r'))
        studies = _testB_set_middle_frame(studies, annotation)
        return studies
    elif annotation_path == '' or annotation_path is None:
        return studies
    else:
        annotation = read_annotation(config, annotation_path)
        counter = _set_middle_frame(studies, annotation)
        return studies, annotation, counter
