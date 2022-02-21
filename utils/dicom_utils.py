from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import SimpleITK as sitk
from typing import Union
from PIL import Image
import torchvision.transforms.functional as tf
import numpy as np
from utils.utils import (
    lazy_property,
    str2tensor,
    unit_vector,
    unit_normal_vector
)
from utils.transforms import (
    random_rotate,
    random_shift,
    random_crop,
    random_intensity,
    random_bias_field,
    random_noise,
    random_hflip,
    resize
)


class DICOM:
    def __init__(self, file_path, DICOM_TAG):
        self.file_path = file_path
        self.error_msg = ''
        self.DICOM_TAG = DICOM_TAG

        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetImageIO('GDCMImageIO')
        reader.SetFileName(file_path)
        try:
            reader.ReadImageInformation()
        except RuntimeError:
            pass

        self.study_uid: str = self._get_meta_data(reader, 'studyUid', self.DICOM_TAG, '')
        self.series_uid: str = self._get_meta_data(reader, 'seriesUid', self.DICOM_TAG, '')
        self.instance_uid: str = self._get_meta_data(reader, 'instanceUid', self.DICOM_TAG, '')
        self.series_description: str = self._get_meta_data(reader, 'seriesDescription', self.DICOM_TAG, '')
        self._pixel_spacing = self._get_meta_data(reader, 'pixelSpacing', self.DICOM_TAG, None)
        self._image_position = self._get_meta_data(reader, 'imagePosition', self.DICOM_TAG, None)
        self._image_orientation = self._get_meta_data(reader, 'imageOrientation', self.DICOM_TAG, None)

        try:
            image = reader.Execute()
            if image.GetNumberOfComponentsPerPixel() == 1:
                image = sitk.RescaleIntensity(image, 0, 255)
                if reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
                    image = sitk.InvertIntensity(image, maximum=255)
                image = sitk.Cast(image, sitk.sitkUInt8)
            img_x = sitk.GetArrayFromImage(image)[0]
            self.image: Image.Image = tf.to_pil_image(img_x)
        except RuntimeError:
            self.image = None

    def _get_meta_data(self, reader: sitk.ImageFileReader, key: str, DICOM_TAG: dict,
                       failed_return: Union[None, str]) -> Union[None, str]:
        try:
            return reader.GetMetaData(DICOM_TAG[key])
        except RuntimeError:
            return failed_return

    @lazy_property
    def pixel_spacing(self):
        if self._pixel_spacing is None:
            return torch.full([2, ], fill_value=np.nan)
        else:
            return str2tensor(self._pixel_spacing)

    @lazy_property
    def image_position(self):
        if self._image_position is None:
            return torch.full([3, ], fill_value=np.nan)
        else:
            return str2tensor(self._image_position)

    @lazy_property
    def image_orientation(self):
        if self._image_orientation is None:
            return torch.full([2, 3], fill_value=np.nan)
        else:
            return unit_vector(str2tensor(self._image_orientation).reshape(2, 3))

    @lazy_property
    def unit_normal_vector(self):
        if self.image_orientation is None:
            return torch.full([3, ], fill_value=np.nan)
        else:
            return unit_normal_vector(self.image_orientation)

    @lazy_property
    def t_type(self):
        if 'T1' in self.series_description.upper():
            return 'T1'
        elif 'T2' in self.series_description.upper():
            return 'T2'
        else:
            return None

    @lazy_property
    def t_info(self):
        return self.series_description.upper()

    @lazy_property
    def plane(self):
        if torch.isnan(self.unit_normal_vector).all():
            return None
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 0., 1.])).abs() > 0.75:
            return 'transverse'
        elif torch.matmul(self.unit_normal_vector, torch.tensor([1., 0., 0.])).abs() > 0.75:
            return 'sagittal'
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 1., 0.])).abs() > 0.75:
            return 'coronal'
        else:
            return None

    @lazy_property
    def mean(self):
        if self.image is None:
            return None
        else:
            return tf.to_tensor(self.image).mean()

    @property
    def size(self):
        if self.image is None:
            return None
        else:
            return self.image.size

    def pixel_coord2human_coord(self, coord: torch.Tensor) -> torch.Tensor:
        return torch.matmul(coord * self.pixel_spacing, self.image_orientation) + self.image_position

    def point_distance(self, human_coord: torch.Tensor) -> torch.Tensor:
        return torch.matmul(human_coord - self.image_position, self.unit_normal_vector).abs()

    def projection(self, human_coord: torch.Tensor) -> torch.Tensor:
        cos = torch.matmul(human_coord - self.image_position, self.image_orientation.transpose(0, 1))
        return (cos / self.pixel_spacing).round()

    def transform(self, trans_dict, pixel_coord: torch.Tensor, tensor=True, isaug=True) -> (
    torch.Tensor, torch.Tensor, dict):
        image = self.image
        pixel_spacing = self.pixel_spacing

        if len(pixel_coord.shape) == 1:
            pixel_coord = pixel_coord.unsqueeze(dim=0)

        details = {'is_hflip': False}
        if isaug and np.random.uniform(0, 1) < trans_dict['rotate_p']:
            image, pixel_coord = random_rotate(image, pixel_coord, trans_dict['max_angel'])
        if isaug and np.random.uniform(0, 1) < trans_dict['shift_p']:
            image, pixel_coord = random_shift(image, pixel_coord, trans_dict['max_shift_ratios'])
        if isaug and np.random.uniform(0, 1) < trans_dict['crop_p']:
            image, pixel_coord = random_crop(image, pixel_coord, trans_dict['max_crop_ratios'])
        if isaug and np.random.uniform(0, 1) < trans_dict['hflip_p']:
            image, pixel_coord = random_hflip(image, pixel_coord)
            details['is_hflip'] = True
        if isaug and np.random.uniform(0, 1) < trans_dict['intensity_p']:
            image = random_intensity(image, trans_dict['max_intensity_ratio'])
        if isaug and np.random.uniform(0, 1) < trans_dict['noise_p']:
            image = random_noise(image, trans_dict['noise_mean'], trans_dict['noise_std'])
        if isaug and np.random.uniform(0, 1) < trans_dict['bias_field_p']:
            image = random_bias_field(image, trans_dict['order'], trans_dict['coefficients_range'])
        # resize
        if trans_dict['size'] is not None:
            image, pixel_coord, pixel_spacing = resize(image, pixel_coord, pixel_spacing, trans_dict['size'])

        if tensor:
            image = tf.to_tensor(image)
        pixel_coord = pixel_coord.round().long()

        return image, pixel_coord, details

    def transverse_transform(self, trans_dict, pixel_coord: torch.Tensor, tensor=True, isaug=True) -> (
    torch.Tensor, torch.Tensor):
        image = self.image
        pixel_spacing = self.pixel_spacing

        if trans_dict['size'] is not None:
            transverse_size = (int(trans_dict['size'][0] / pixel_spacing[0].item()),
                               int(trans_dict['size'][1] / pixel_spacing[1].item()))
            image = tf.crop(image,
                            int(pixel_coord[1].item() - transverse_size[0] // 2),
                            int(pixel_coord[0].item() - transverse_size[1] // 2),
                            transverse_size[0],
                            transverse_size[1])

        if len(pixel_coord.shape) == 1:
            pixel_coord = pixel_coord.unsqueeze(dim=0)

        if isaug and np.random.uniform(0, 1) < trans_dict['rotate_p']:
            image, pixel_coord = random_rotate(image, pixel_coord, trans_dict['max_angel'])
        if isaug and np.random.uniform(0, 1) < trans_dict['shift_p']:
            image, pixel_coord = random_shift(image, pixel_coord, trans_dict['max_shift_ratios'])
        if isaug and np.random.uniform(0, 1) < trans_dict['crop_p']:
            image, pixel_coord = random_crop(image, pixel_coord, trans_dict['max_crop_ratios'])
        if isaug and np.random.uniform(0, 1) < trans_dict['hflip_p']:
            image, pixel_coord = random_hflip(image, pixel_coord)
        if isaug and np.random.uniform(0, 1) < trans_dict['intensity_p']:
            image = random_intensity(image, trans_dict['max_intensity_ratio'])
        if isaug and np.random.uniform(0, 1) < trans_dict['noise_p']:
            image = random_noise(image, trans_dict['noise_mean'], trans_dict['noise_std'])
        if isaug and np.random.uniform(0, 1) < trans_dict['bias_field_p']:
            image = random_bias_field(image, trans_dict['order'], trans_dict['coefficients_range'])
        # resize
        if trans_dict['size'] is not None:
            image, pixel_coord, pixel_spacing = resize(image, pixel_coord, pixel_spacing, trans_dict['size'])

        if tensor:
            image = tf.to_tensor(image)
        pixel_coord = pixel_coord.round().long()

        return image, pixel_coord