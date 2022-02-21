from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as tf
import numpy as np
import math


def random_rotate(image: Image.Image, pixel_coord: torch.Tensor, max_angel: int) \
        -> (Image.Image, torch.Tensor):
    angel = np.random.randint(-1 * max_angel, max_angel)
    center = torch.tensor(image.size, dtype=torch.float32) / 2
    image = tf.rotate(image, angel, fill=(0,) * (3 if len(np.shape(image)) == 3 else 1))
    if angel != 0:
        angel = angel * math.pi / 180
        while len(center.shape) < len(pixel_coord.shape):
            center = center.unsqueeze(0)
        cos = math.cos(angel)
        sin = math.sin(angel)
        rotate_mat = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32, device=pixel_coord.device)
        output = pixel_coord - center
        output = torch.matmul(output, rotate_mat)
        pixel_coord = output + center
    return image, pixel_coord


def random_shift(image: Image.Image, pixel_coord: torch.Tensor, max_shift_ratios: tuple) \
        -> (Image.Image, torch.Tensor):
    # max_shift_ratios: (height_ratio, width_ratio)
    height, width = image.size[1], image.size[0]
    center = (height // 2, width // 2)
    top_shift = int(height * np.random.uniform(-1 * max_shift_ratios[0], max_shift_ratios[0]))
    lef_shift = int(width * np.random.uniform(-1 * max_shift_ratios[1], max_shift_ratios[1]))
    image = tf.pad(image, (width // 2, height // 2, width // 2, height // 2))
    center = (center[0] + height // 2 + top_shift, center[1] + width // 2 + lef_shift)
    image = tf.crop(image, center[0] - height // 2, center[1] - width // 2, height, width)
    pixel_coord[:, 0] -= lef_shift
    pixel_coord[:, 1] -= top_shift
    return image, pixel_coord


def random_crop(image: Image.Image, pixel_coord: torch.Tensor, max_crop_ratios: tuple) \
        -> (Image.Image, torch.Tensor):
    # max_crop_ratios: (height_ratio, width_ratio)
    height, width = image.size[1], image.size[0]
    center = (height // 2, width // 2)
    new_height = int(height * (1. + np.random.uniform(-1 * max_crop_ratios[0], max_crop_ratios[0])))
    new_width = int(width * (1. + np.random.uniform(-1 * max_crop_ratios[1], max_crop_ratios[1])))
    image = tf.pad(image, (width // 2, height // 2, width // 2, height // 2))
    center = (center[0] + height // 2, center[1] + width // 2)
    image = tf.crop(image, center[0] - new_height // 2, center[1] - new_width // 2, new_height, new_width)
    pixel_coord[:, 0] += (new_width - width) // 2
    pixel_coord[:, 1] += (new_height - height) // 2
    return image, pixel_coord


def random_intensity(image: Image.Image, max_intensity_ratio: float) \
        -> Image.Image:
    np_image = np.asarray(image)
    np_image = np_image * (1. + np.random.uniform(-1 * max_intensity_ratio, max_intensity_ratio))
    np_image = np_image.clip(0, 255)
    image = Image.fromarray(np.uint8(np_image))
    return image


def random_hflip(image: Image.Image, pixel_coord: torch.Tensor) -> (Image.Image, torch.Tensor):
    height, width = image.size[1], image.size[0]
    image = tf.hflip(image)
    pixel_coord[:, 0] = width - pixel_coord[:, 0] + 1
    return image, pixel_coord


def resize(image: Image.Image, pixel_coord: torch.Tensor, pixel_spacing: torch.Tensor, size: Tuple[int, int]) \
        -> (Image.Image, torch.Tensor, torch.Tensor):
    height_ratio = size[0] / image.size[1]
    width_ratio = size[1] / image.size[0]
    ratio = torch.tensor([width_ratio, height_ratio])
    image = tf.resize(image, size)
    pixel_coord = pixel_coord * ratio
    pixel_spacing = pixel_spacing / ratio
    return image, pixel_coord, pixel_spacing


def random_bias_field(image: Image.Image, order: int = 3, coefficients_range: list = [-0.5, 0.5]) \
        -> Image.Image:
    image = np.asarray(image)
    shape = np.asarray(image.shape[:2])
    half_shape = shape / 2

    ranges = [np.arange(-n, n) for n in half_shape]

    bias_field = np.zeros(shape)
    y_mesh, x_mesh = np.asarray(np.meshgrid(*ranges))

    y_mesh /= y_mesh.max()
    x_mesh /= x_mesh.max()

    random_coefficients = []
    for y_order in range(0, order + 1):
        for x_order in range(0, order + 1 - y_order):
            number = np.random.uniform(*coefficients_range)
            random_coefficients.append(number)

    i = 0
    for y_order in range(order + 1):
        for x_order in range(order + 1 - y_order):
            random_coefficient = random_coefficients[i]
            new_map = (random_coefficient
                       * y_mesh ** y_order
                       * x_mesh ** x_order)
            bias_field += np.transpose(new_map, (1, 0))
            i += 1
    bias_field = np.exp(bias_field).astype(np.float32)
    if len(np.shape(image)) == 3:
        image = image * bias_field[..., np.newaxis]
    else:
        image = image * bias_field
    image = Image.fromarray(np.uint8(image))
    return image

def random_noise(image: Image.Image, mean: list = [0, 5], std: list = [0, 2]) \
        -> Image.Image:
    image = np.asarray(image)
    noise = np.random.randn(*image.shape) * np.random.uniform(*std) + np.random.uniform(*mean)
    image = Image.fromarray(np.uint8(image + noise))
    return image