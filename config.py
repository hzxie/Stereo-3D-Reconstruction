# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.LEFT_RENDERING_PATH   = '/home/hzxie/Datasets/ShapeNet/ShapeNetStereoRendering/%s/%s/render_%02d_l.png'
__C.DATASETS.SHAPENET.RIGHT_RENDERING_PATH  = '/home/hzxie/Datasets/ShapeNet/ShapeNetStereoRendering/%s/%s/render_%02d_r.png'
__C.DATASETS.SHAPENET.LEFT_DISP_PATH        = '/home/hzxie/Datasets/ShapeNet/ShapeNetStereoRendering/%s/%s/disp_%02d_l.exr'
__C.DATASETS.SHAPENET.RIGHT_DISP_PATH       = '/home/hzxie/Datasets/ShapeNet/ShapeNetStereoRendering/%s/%s/disp_%02d_r.exr'
__C.DATASETS.SHAPENET.VOLUME_PATH           = '/home/hzxie/Datasets/ShapeNet/ShapeNetVox32/%s/%s.mat'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.DATASET_NAME                    = 'ShapeNet'
__C.DATASET.IMG_MEAN                        = [0.5, 0.5, 0.5]
__C.DATASET.IMG_STD                         = [255, 255, 255]
__C.DATASET.DISP_NORM_FACTOR                = 20

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 137       # Image width for input
__C.CONST.IMG_H                             = 137       # Image height for input
__C.CONST.IMG_C                             = 4         # Image channels for input
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 20
__C.CONST.N_VIEWS                           = 24
__C.CONST.CROP_IMG_W                        = 210
__C.CONST.CROP_IMG_H                        = 210
__C.CONST.CROP_IMG_C                        = 4

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 500
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.DISPNET_LEARNING_RATE             = 1e-4
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-4
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-4
__C.TRAIN.FUSNET_LEARNING_RATE              = 1e-4
__C.TRAIN.DISPNET_LR_MILESTONES             = [300]
__C.TRAIN.ENCODER_LR_MILESTONES             = [300]
__C.TRAIN.DECODER_LR_MILESTONES             = [300]
__C.TRAIN.FUSNET_LR_MILESTONES              = [300]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 25            # weights will be overwritten every save_freq epoch

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
