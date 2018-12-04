# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test import test_net
from models.dispnet import DispNet
from models.recnet import RecNet


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W, cfg.CONST.CROP_IMG_C
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomBackground(cfg.DIR.RANDOM_BG_PATH),
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.Normalize(cfg.DATASET.IMG_MEAN, cfg.DATASET.IMG_STD, cfg.DATASET.DISP_NORM_FACTOR),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomBackground(cfg.DIR.RANDOM_BG_PATH),
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.Normalize(cfg.DATASET.IMG_MEAN, cfg.DATASET.IMG_STD, cfg.DATASET.DISP_NORM_FACTOR),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME](cfg)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER,
        pin_memory=True,
        shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS, val_transforms),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False)

    # Set up networks
    dispnet = DispNet(cfg)
    recnet = RecNet(cfg)
    print('[DEBUG] %s Parameters in DispNet: %d.' % (dt.now(), utils.network_utils.count_parameters(dispnet)))
    print('[DEBUG] %s Parameters in RecNet: %d.' % (dt.now(), utils.network_utils.count_parameters(recnet)))

    # Initialize weights of networks
    dispnet.apply(utils.network_utils.init_weights)
    recnet.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        dispnet_solver = torch.optim.Adam(
            filter(lambda p: p.requires_grad, dispnet.parameters()),
            lr=cfg.TRAIN.DISPNET_LEARNING_RATE,
            betas=cfg.TRAIN.BETAS)
        recnet_solver = torch.optim.Adam(
            filter(lambda p: p.requires_grad, recnet.parameters()),
            lr=cfg.TRAIN.RECNET_LEARNING_RATE,
            betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        dispnet_solver = torch.optim.SGD(
            filter(lambda p: p.requires_grad, dispnet.parameters()),
            lr=cfg.TRAIN.DISPNET_LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM)
        recnet_solver = torch.optim.SGD(
            filter(lambda p: p.requires_grad, recnet.parameters()),
            lr=cfg.TRAIN.RECNET_LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    dispnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        dispnet_solver, milestones=cfg.TRAIN.DISPNET_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    recnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        recnet_solver, milestones=cfg.TRAIN.RECNET_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        dispnet = torch.nn.DataParallel(dispnet).cuda()
        recnet = torch.nn.DataParallel(recnet).cuda()

    # Set up loss functions
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        dispnet.load_state_dict(checkpoint['dispnet_state_dict'])
        dispnet_solver.load_state_dict(checkpoint['dispnet_solver_state_dict'])
        recnet.load_state_dict(checkpoint['recnet_state_dict'])
        recnet_solver.load_state_dict(checkpoint['recnet_solver_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' \
                 % (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        disparity_losses = utils.network_utils.AverageMeter()
        voxel_losses = utils.network_utils.AverageMeter()

        # Adjust learning rate
        dispnet_lr_scheduler.step()
        recnet_lr_scheduler.step()

        # switch models to training mode
        dispnet.train()
        recnet.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_ids, sample_names, left_rgb_images, right_rgb_images, left_disp_images,
                        right_disp_images, ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            n_samples = len(ground_truth_volumes)
            # Ignore imcomplete batches at the end of each epoch
            if not n_samples == cfg.CONST.BATCH_SIZE:
                continue

            # Put it to GPU if CUDA is available
            left_rgb_images = utils.network_utils.var_or_cuda(left_rgb_images)
            right_rgb_images = utils.network_utils.var_or_cuda(right_rgb_images)
            left_disp_images = utils.network_utils.var_or_cuda(left_disp_images)
            right_disp_images = utils.network_utils.var_or_cuda(right_disp_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)

            # Train the DispNet and RecNet
            left_disp_estimated, right_disp_estimated = dispnet(left_rgb_images, right_rgb_images)
            left_rgbd_images = torch.cat((left_rgb_images, left_disp_estimated), dim=1)
            right_rgbd_images = torch.cat((right_rgb_images, right_disp_estimated), dim=1)

            left_generated_volumes = recnet(left_rgbd_images)
            right_generated_volumes = recnet(right_rgbd_images)
            # TODO: Use a better method to fuse two Stereo volumes
            generated_volumes = torch.cat((left_generated_volumes, right_generated_volumes), dim=1)
            generated_volumes = torch.mean(generated_volumes, dim=1)

            # Calculate losses for disp estimation and voxel reconstruction
            disparity_loss = mse_loss(left_disp_estimated, left_disp_images) + \
                             mse_loss(right_disp_estimated, right_disp_images)
            voxel_loss = bce_loss(generated_volumes, ground_truth_volumes)

            # Gradient decent
            dispnet.zero_grad()
            recnet.zero_grad()
            disparity_loss.backward(retain_graph=True)
            voxel_loss.backward()
            dispnet_solver.step()
            recnet_solver.step()

            # Append loss to average metrics
            disparity_losses.update(disparity_loss.item())
            voxel_losses.update(voxel_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('DispNet/BatchLoss', disparity_loss.item(), n_itr)
            train_writer.add_scalar('RecNet/BatchLoss', voxel_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print('[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) DLoss = %.4f VLoss = %.4f' % \
                (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, \
                    batch_time.val, data_time.val, disparity_loss.item(), voxel_loss.item()))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('DispNet/EpochLoss', disparity_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('RecNet/EpochLoss', voxel_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) DLoss = %.4f VLoss = %.4f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, disparity_losses.avg, voxel_losses.avg))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, dispnet, recnet)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, \
                    os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                    epoch_idx + 1, dispnet, dispnet_solver, recnet, recnet_solver, \
                    best_iou, best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, \
                    os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                    epoch_idx + 1, dispnet, dispnet_solver, recnet, recnet_solver, \
                    best_iou, best_epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
