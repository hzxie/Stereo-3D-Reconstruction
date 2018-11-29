# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import sys
import torch.utils.data.dataset

from datetime import datetime as dt
from enum import Enum, unique

import utils.binvox_rw


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""

    def __init__(self, file_list_with_metadata, transforms=None):
        self.file_list = file_list_with_metadata
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, left_rgb_image, right_rgb_image, \
            left_depth_image, right_depth_image, volume = self.get_datum(idx)

        if self.transforms:
            left_rgb_image, right_rgb_image, left_depth_image, right_depth_image = self.transforms(
                left_rgb_image, right_rgb_image, left_depth_image, right_depth_image)

        return taxonomy_name, sample_name, left_rgb_image, right_rgb_image, left_depth_image, right_depth_image, volume

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        left_rgb_images_file_path = self.file_list[idx]['left_rgb_images']
        right_rgb_images_file_path = self.file_list[idx]['right_rgb_images']
        left_depth_images_file_path = self.file_list[idx]['left_depth_images']
        right_depth_images_file_path = self.file_list[idx]['right_depth_images']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        selected_index = random.choice(range(len(left_rgb_images_file_path)))
        left_rgb_image_file_path = left_rgb_images_file_path[selected_index]
        right_rgb_image_file_path = right_rgb_images_file_path[selected_index]
        left_depth_image_file_path = left_depth_images_file_path[selected_index]
        right_depth_image_file_path = right_depth_images_file_path[selected_index]

        left_rgb_image = cv2.imread(left_rgb_image_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        right_rgb_image = cv2.imread(right_rgb_image_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        left_depth_image = cv2.imread(left_depth_image_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        right_depth_image = cv2.imread(right_depth_image_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Get data of voxel
        volume = scipy.io.loadmat(volume_path)
        if not volume:
            print('[FATAL] %s Failed to get volume data from file %s' % (dt.now(), volume_path))
            sys.exit(2)

        volume = volume['Volume'].astype(np.float32)
        return taxonomy_name, sample_name, left_rgb_image, right_rgb_image, left_depth_image, right_depth_image, volume


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.left_rgb_image_path_template = cfg.DATASETS.SHAPENET.LEFT_RENDERING_PATH
        self.right_rgb_image_path_template = cfg.DATASETS.SHAPENET.RIGHT_RENDERING_PATH
        self.left_depth_image_path_template = cfg.DATASETS.SHAPENET.LEFT_DEPTH_PATH
        self.right_depth_image_path_template = cfg.DATASETS.SHAPENET.RIGHT_DEPTH_PATH
        self.volume_path_template = cfg.DATASETS.SHAPENET.VOLUME_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, total_views, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']
            print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' % (dt.now(), taxonomy['taxonomy_id'],
                                                                              taxonomy['taxonomy_name']))

            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(taxonomy_folder_name, samples, total_views))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return ShapeNetDataset(files, transforms)

    def get_files_of_taxonomy(self, taxonomy_folder_name, samples, total_views):
        files_of_taxonomy = []
        n_samples = len(samples)

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of voxels
            volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(volume_file_path):
                print('[WARN] %s Ignore sample %s/%s since voxel file not exists.' % (dt.now(), taxonomy_folder_name,
                                                                                      sample_name))
                continue

            # Get file list of rendering images
            image_indexes = range(total_views)
            left_rgb_images_file_path = []
            right_rgb_images_file_path = []
            left_depth_images_file_path = []
            right_depth_images_file_path = []
            for image_idx in image_indexes:
                left_rgb_file_path = self.left_rgb_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                right_rgb_file_path = self.right_rgb_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                left_depth_file_path = self.left_depth_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                right_depth_file_path = self.right_depth_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                # if not os.path.exists(left_rgb_file_path):
                #     print('[WARN] %s Ignore rendering image of sample %s/%s/%d since file not exists.' % \
                #         (dt.now(), taxonomy_folder_name, sample_name, image_idx))
                #     continue
                left_rgb_images_file_path.append(left_rgb_file_path)
                right_rgb_images_file_path.append(right_rgb_file_path)
                left_depth_images_file_path.append(left_depth_file_path)
                right_depth_images_file_path.append(right_depth_file_path)

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'left_rgb_images': left_rgb_images_file_path,
                'right_rgb_images': right_rgb_images_file_path,
                'left_depth_images': left_depth_images_file_path,
                'right_depth_images': right_depth_images_file_path,
                'volume': volume_file_path
            })

            # Report the progress of reading dataset
            # if sample_idx % 500 == 499 or sample_idx == n_samples - 1:
            #     print('[INFO] %s Collecting %d of %d' % (dt.now(), sample_idx + 1, n_samples))

        return files_of_taxonomy


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
}
