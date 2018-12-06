# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torchvision.transforms

from PIL import Image


class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.RandomBackground(),
    >>>     transforms.CenterCrop(127, 127, 3),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image):
        for t in self.transforms:
            left_rgb_image, right_rgb_image, left_disp_image, right_disp_image = t(
                left_rgb_image, right_rgb_image, left_disp_image, right_disp_image)

        return left_rgb_image, right_rgb_image, left_disp_image, right_disp_image


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image):
        left_rgb_image = self.to_tensor(left_rgb_image.copy())
        right_rgb_image = self.to_tensor(right_rgb_image.copy())
        left_disp_image = self.to_tensor(left_disp_image.copy())
        right_disp_image = self.to_tensor(right_disp_image.copy())

        return left_rgb_image, right_rgb_image, left_disp_image, right_disp_image

    def to_tensor(self, img):
        arr = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(arr)
        return tensor.float()


class Normalize(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image):
        left_rgb_image = self.normalize(left_rgb_image)
        right_rgb_image = self.normalize(right_rgb_image)

        return left_rgb_image, right_rgb_image, left_disp_image, right_disp_image

    def normalize(self, img):
        img /= self.img_std
        img -= self.img_mean
        img *= 2
        return img


class RandomPermuteRGB(object):
    def __call__(self, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image):
        rgb_permutation = np.random.permutation(3)
        left_rgb_image = self.random_permute_rgb(left_rgb_image, rgb_permutation)
        right_rgb_image = self.random_permute_rgb(right_rgb_image, rgb_permutation)

        return left_rgb_image, right_rgb_image, left_disp_image, right_disp_image

    def random_permute_rgb(self, img, rgb_permutation):
        return img[..., rgb_permutation]


class CenterCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image):
        img_width, img_height, _ = left_rgb_image.shape

        x_left = (img_width - self.crop_size_w) * 0.5
        x_right = x_left + self.crop_size_w
        y_top = (img_height - self.crop_size_h) * 0.5
        y_bottom = y_top + self.crop_size_h

        left_rgb_image = self.center_crop(left_rgb_image, x_left, x_right, y_top, y_bottom)
        right_rgb_image = self.center_crop(right_rgb_image, x_left, x_right, y_top, y_bottom)
        left_disp_image = self.center_crop(left_disp_image, x_left, x_right, y_top, y_bottom)
        right_disp_image = self.center_crop(right_disp_image, x_left, x_right, y_top, y_bottom)

        return left_rgb_image, right_rgb_image, \
               np.expand_dims(left_disp_image, axis=2), np.expand_dims(right_disp_image, axis=2)

    def center_crop(self, img, x_left, x_right, y_top, y_bottom):
        return cv2.resize(img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))


class RandomCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]
        self.crop_size_c = crop_size[2]

    def __call__(self, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image):
        img_width, img_height, _ = left_rgb_image.shape
        
        x_left = (img_width - self.crop_size_w) * random.random()
        x_right = x_left + self.crop_size_w
        y_top = (img_height - self.crop_size_h) * random.random()
        y_bottom = y_top + self.crop_size_h

        left_rgb_image = self.random_crop(left_rgb_image, x_left, x_right, y_top, y_bottom)
        right_rgb_image = self.random_crop(right_rgb_image, x_left, x_right, y_top, y_bottom)
        left_disp_image = self.random_crop(left_disp_image, x_left, x_right, y_top, y_bottom)
        right_disp_image = self.random_crop(right_disp_image, x_left, x_right, y_top, y_bottom)

        return left_rgb_image, right_rgb_image, \
               np.expand_dims(left_disp_image, axis=2), np.expand_dims(right_disp_image, axis=2)

    def random_crop(self, img, x_left, x_right, y_top, y_bottom):
        return cv2.resize(img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))


class RandomFlip(object):
    def __call__(self, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image):
        if random.random() > 0.5:
            left_rgb_image = np.fliplr(left_rgb_image)
            right_rgb_image = np.fliplr(right_rgb_image)
            left_disp_image = np.fliplr(left_disp_image)
            right_disp_image = np.fliplr(right_disp_image)

        return left_rgb_image, right_rgb_image, left_disp_image, right_disp_image


class RandomBackground(object):
    def __init__(self, random_bg_color_range):
        self.random_bg_color_range = random_bg_color_range

    def __call__(self, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image):
        img_height, img_width, img_channels = left_rgb_image.shape
        if not img_channels == 4:
            return left_rgb_image, right_rgb_image, left_disp_image, right_disp_image

        # If the image has the alpha channel, add the background
        r, g, b = [
            np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1)
            for i in range(3)
        ]
        left_rgb_image = self.random_background(left_rgb_image, r, g, b)
        right_rgb_image = self.random_background(right_rgb_image, r, g, b)

        return left_rgb_image, right_rgb_image, left_disp_image, right_disp_image

    def random_background(self, img, r, g, b):
        alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
        img = img[:, :, :3]
        bg_color = np.array([[[r, g, b]]])
        img = alpha * bg_color + (1 - alpha) * img

        return img
