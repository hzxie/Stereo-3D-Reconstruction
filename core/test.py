# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from models.recnet import RecNet

def test_net(cfg, epoch_idx=-1, output_dir=None, test_data_loader=None, \
        test_writer=None, depnet=None, recnet=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.DATASET_NAME.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W, cfg.CONST.CROP_IMG_C
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME](cfg)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS, test_transforms),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False)

    # Set up networks
    if recnet is None:
        recnet = RecNet(cfg)

        if torch.cuda.is_available():
            recnet = torch.nn.DataParallel(recnet).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        recnet.load_state_dict(checkpoint['recnet_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    recnet_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    recnet.eval()

    for sample_idx, (taxonomy_id, sample_name, left_rgb_image, right_rgb_image, left_depth_image, right_depth_image,
                     ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            left_rgb_image = utils.network_utils.var_or_cuda(left_rgb_image)
            right_rgb_image = utils.network_utils.var_or_cuda(right_rgb_image)
            left_depth_image = utils.network_utils.var_or_cuda(left_depth_image)
            right_depth_image = utils.network_utils.var_or_cuda(right_depth_image)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Train the DepNet and RecNet
            # TODO: Use a DepNet to estimate depth
            left_depth_estimated = left_depth_image
            right_depth_estimated = right_depth_image
            left_rgbd_image = torch.cat((left_rgb_image, left_depth_estimated), dim=1)
            right_rgbd_image = torch.cat((right_rgb_image, right_depth_estimated), dim=1)

            left_generated_volume = recnet(left_rgbd_image)
            right_generated_volume = recnet(right_rgbd_image)
            # TODO: Use a better method to fuse two Stereo volumes
            generated_volume = torch.cat((left_generated_volume, right_generated_volume), dim=1)
            generated_volume = torch.mean(generated_volume, dim=1)

            recnet_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if not taxonomy_id in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Append generated volumes to TensorBoard
            if sample_idx < 3:
                img_dir = output_dir % 'images'
                gtv = ground_truth_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_voxel_views(gtv, os.path.join(img_dir, 'test'), epoch_idx)
                test_writer.add_image('Ground Truth Voxels', rendering_views, epoch_idx)
                gv = generated_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_voxel_views(gv, os.path.join(img_dir, 'test'), epoch_idx)
                test_writer.add_image('Reconstructed Voxels', rendering_views, epoch_idx)

            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s RLoss = %.4f IoU = %s' % \
                (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, recnet_loss.item(), \
                    ['%.4f' % si for si in sample_iou]))

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if not test_writer is None:
        test_writer.add_scalar('RecNet/EpochLoss', recnet_losses.avg, epoch_idx)
        test_writer.add_scalar('RecNet/IoU', max_iou, epoch_idx)

    return max_iou
