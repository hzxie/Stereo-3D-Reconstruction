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

import utils.visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from extensions.chamfer_dist import ChamferDistance
from models.corrnet import CorrelationNet
from models.dispnet import DispNet
from models.encoder import Encoder
from models.decoder import Decoder

def test_net(cfg, epoch_idx=-1, output_dir=None, test_data_loader=None, \
        test_writer=None, dispnet=None, encoder=None, decoder=None, corrnet=None):
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
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.Normalize(cfg.DATASET.IMG_MEAN, cfg.DATASET.IMG_STD),
            utils.data_transforms.ToTensor(),
            utils.data_transforms.RandomSamplePoints(cfg.NETWORK.N_POINTS),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME](cfg)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS,
                                               test_transforms),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False)

    # Set up networks
    if dispnet is None or encoder is None or decoder is None or corrnet is None:
        dispnet = DispNet(cfg)
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        corrnet = CorrelationNet(cfg)

        if torch.cuda.is_available():
            dispnet = torch.nn.DataParallel(dispnet).cuda()
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            corrnet = torch.nn.DataParallel(corrnet).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        dispnet.load_state_dict(checkpoint['dispnet_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        corrnet.load_state_dict(checkpoint['corrnet_state_dict'])

    # Set up loss functions
    mse_loss = torch.nn.MSELoss()
    chamfer_distance = ChamferDistance()

    # Testing loop
    n_samples = len(test_data_loader)
    test_cd = dict()
    disparity_losses = utils.network_utils.AverageMeter()
    pt_cloud_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    dispnet.eval()
    encoder.eval()
    decoder.eval()
    corrnet.eval()

    for sample_idx, (taxonomy_id, sample_name, left_rgb_image, right_rgb_image, left_disp_image, right_disp_image,
                     ground_ptcloud) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            left_rgb_image = utils.network_utils.var_or_cuda(left_rgb_image)
            right_rgb_image = utils.network_utils.var_or_cuda(right_rgb_image)
            left_disp_image = utils.network_utils.var_or_cuda(left_disp_image)
            right_disp_image = utils.network_utils.var_or_cuda(right_disp_image)
            ground_ptcloud = utils.network_utils.var_or_cuda(ground_ptcloud)

            # Train the DispNet and RecNet
            left_disp_estimated, right_disp_estimated, disp_features = dispnet(left_rgb_image, right_rgb_image)
            left_rgbd_image = torch.cat((left_rgb_image, left_disp_estimated), dim=1)
            right_rgbd_image = torch.cat((right_rgb_image, right_disp_estimated), dim=1)

            left_img_features, left_ll_features = encoder(left_rgbd_image)
            right_img_features, right_ll_features = encoder(right_rgbd_image)
            corr_features = corrnet(left_ll_features, right_ll_features)
            generated_ptcloud = decoder(left_img_features, right_img_features, corr_features)

            # Calculate losses for disp estimation and voxel reconstruction
            disparity_loss = mse_loss(left_disp_estimated, left_disp_image) + \
                             mse_loss(right_disp_estimated, right_disp_image)
            pt_cloud_loss = chamfer_distance(generated_ptcloud, ground_ptcloud) * 1000

            # Append loss and accuracy to average metrics
            disparity_losses.update(disparity_loss.item())
            pt_cloud_losses.update(pt_cloud_loss.item())

            # Chamfer Distance per taxonomy
            if not taxonomy_id in test_cd:
                test_cd[taxonomy_id] = {'n_samples': 0, 'cd': []}

            test_cd[taxonomy_id]['n_samples'] += 1
            test_cd[taxonomy_id]['cd'].append(pt_cloud_loss.item())

            # Append generated volumes to TensorBoard
            if output_dir and sample_idx < 3:
                img_dir = output_dir % 'images'
                # Disparity Map Visualization
                test_writer.add_image('Test Sample#%02d/Left Disparity Estimated' % sample_idx,
                                      left_disp_estimated / cfg.DATASET.MAX_DISP_VALUE, epoch_idx)
                test_writer.add_image('Test Sample#%02d/Left Disparity GroundTruth' % sample_idx,
                                      left_disp_image / cfg.DATASET.MAX_DISP_VALUE, epoch_idx)
                test_writer.add_image('Test Sample#%02d/Right Disparity Estimated' % sample_idx,
                                      right_disp_estimated / cfg.DATASET.MAX_DISP_VALUE, epoch_idx)
                test_writer.add_image('Test Sample#%02d/Right Disparity GroundTruth' % sample_idx,
                                      right_disp_image / cfg.DATASET.MAX_DISP_VALUE, epoch_idx)
                # Point Cloud Visualization
                gpt = generated_ptcloud.squeeze().cpu().numpy()
                rendering_views = utils.visualization.get_ptcloud_views(gpt, os.path.join(img_dir, 'test'), epoch_idx)
                test_writer.add_image('Test Sample#%02d/Points Reconstructed' % sample_idx, rendering_views, epoch_idx)
                gt = ground_ptcloud.squeeze().cpu().numpy()
                rendering_views = utils.visualization.get_ptcloud_views(gt, os.path.join(img_dir, 'test'), epoch_idx)
                test_writer.add_image('Test Sample#%02d/Points GroundTruth' % sample_idx, rendering_views, epoch_idx)

            # Print sample loss and Chamfer Distance
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s DLoss = %.4f PTLoss = %.4f' % \
                (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, disparity_loss.item(),
                    pt_cloud_loss.item()))

    # Output testing results
    mean_cd = []
    for taxonomy_id in test_cd:
        mean_cd.append(np.mean(test_cd[taxonomy_id]['cd']) * test_cd[taxonomy_id]['n_samples'])
    mean_cd = np.sum(mean_cd) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('ChamferDistance')
    # Print body
    for taxonomy_id in test_cd:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_cd[taxonomy_id]['n_samples'], end='\t')
        print('%.4f' % np.mean(test_cd[taxonomy_id]['cd']), end='\t')
        print()
    # Print mean Chamfer Distance
    print('Overall\t\t\t%.4f' % mean_cd)

    # Add testing results to TensorBoard
    if not test_writer is None:
        test_writer.add_scalar('DispNet/EpochLoss', disparity_losses.avg, epoch_idx)
        test_writer.add_scalar('RecNet/EpochLoss', pt_cloud_losses.avg, epoch_idx)

    return mean_cd
