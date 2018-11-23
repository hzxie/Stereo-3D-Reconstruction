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

    def __init__(self, file_list_with_metadata, n_rendering_views, transforms=None):
        self.file_list = file_list_with_metadata
        self.transforms = transforms
        self.n_rendering_views = n_rendering_views

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, voxel = self.get_datum(idx)

        if self.transforms:
            rendering_images, voxel = self.transforms(rendering_images, voxel)

        return taxonomy_name, sample_name, rendering_images, voxel

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        depth_image_paths = self.file_list[idx]['depth_images']
        voxel_path = self.file_list[idx]['voxel']

        # Get data of rendering images
        rendering_images = []
        selected_image_indexes = random.sample(range(len(rendering_image_paths)), self.n_rendering_views)
        selected_rendering_image_paths = [rendering_image_paths[i] for i in selected_image_indexes]
        selected_depth_image_paths = [depth_image_paths[i] for i in selected_image_indexes]

        for i in range(len(selected_image_indexes)):
            rendering_image = cv2.imread(selected_rendering_image_paths[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_image = cv2.imread(selected_depth_image_paths[i], cv2.IMREAD_UNCHANGED).astype(np.float32)

            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                      (dt.now(), rendering_image_path))
                sys.exit(2)
            # Replace alpha channel with depth
            rendering_image[:, :, 3] = depth_image
            rendering_images.append(rendering_image)

        # Get data of voxel
        voxel = scipy.io.loadmat(voxel_path)
        if not voxel:
            print('[FATAL] %s Failed to get voxel data from file %s' % (dt.now(), voxel_path))
            sys.exit(2)

        voxel = voxel['Volume'].astype(np.float32)
        return taxonomy_name, sample_name, np.asarray(rendering_images), voxel


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.depth_image_path_template = cfg.DATASETS.SHAPENET.DEPTH_PATH
        self.voxel_path_template = cfg.DATASETS.SHAPENET.VOXEL_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, total_views, n_rendering_views, transforms=None):
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
        return ShapeNetDataset(files, n_rendering_views, transforms)

    def get_files_of_taxonomy(self, taxonomy_folder_name, samples, total_views):
        files_of_taxonomy = []
        n_samples = len(samples)

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of voxels
            voxel_file_path = self.voxel_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(voxel_file_path):
                print('[WARN] %s Ignore sample %s/%s since voxel file not exists.' % (dt.now(), taxonomy_folder_name,
                                                                                      sample_name))
                continue

            # Get file list of rendering images
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            depth_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                depth_file_path = self.depth_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                # if not os.path.exists(img_file_path):
                #     print('[WARN] %s Ignore rendering image of sample %s/%s/%d since file not exists.' % \
                #         (dt.now(), taxonomy_folder_name, sample_name, image_idx))
                #     continue
                rendering_images_file_path.append(img_file_path)
                depth_images_file_path.append(depth_file_path)

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
                'depth_images': depth_images_file_path,
                'voxel': voxel_file_path,
            })

            # Report the progress of reading dataset
            # if sample_idx % 500 == 499 or sample_idx == n_samples - 1:
            #     print('[INFO] %s Collecting %d of %d' % (dt.now(), sample_idx + 1, n_samples))

        return files_of_taxonomy


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
}
