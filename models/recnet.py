# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class RecNet(torch.nn.Module):
    def __init__(self, cfg):
        super(RecNet, self).__init__()
        self.cfg = cfg

        # Encoder
        self.conv1a = torch.nn.Sequential(
            torch.nn.Conv2d(4, 96, kernel_size=7, padding=3),
            torch.nn.BatchNorm2d(96),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv1b = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        
        self.conv2a = torch.nn.Sequential(
            torch.nn.Conv2d(96, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv2b = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv2c = torch.nn.Conv2d(96, 128, kernel_size=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3a = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv3b = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv3c = torch.nn.Conv2d(128, 256, kernel_size=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv4a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv4b = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv5a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv5b = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv5c = torch.nn.Conv2d(256, 256, kernel_size=1)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv6a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv6b = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.pool6 = torch.nn.MaxPool2d(kernel_size=2)
        
        self.fc7 = torch.nn.Linear(1024, 8192)

        # Decoder
        self.unpool1a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, bias=False, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool1b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool1c = torch.nn.ConvTranspose3d(128, 128, kernel_size=1, stride=2, bias=False, output_padding=1)

        self.unpool2a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, bias=False, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool2b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool2c = torch.nn.ConvTranspose3d(128, 128, kernel_size=1, stride=2, bias=False, output_padding=1)

        self.unpool3a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, bias=False, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool3b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool3c = torch.nn.ConvTranspose3d(128, 64, kernel_size=1, stride=2, bias=False, output_padding=1)

        self.unpool4a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool4b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 32, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool4c = torch.nn.ConvTranspose3d(64, 32, kernel_size=1, stride=1, bias=False)

        self.unpool5a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool5b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 8, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool5c = torch.nn.ConvTranspose3d(32, 8, kernel_size=1, stride=1, bias=False)

        self.unpool6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, rgbd_images):
        # print(rgbd_images.size())  # torch.Size([batch_size, 4, 137, 137])
        features = self.conv1a(rgbd_images.view(-1, self.cfg.CONST.IMG_C, self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W))
        # print(features.size())    # torch.Size([batch_size, 96, 137, 137])
        features = self.conv1b(features)
        # print(features.size())    # torch.Size([batch_size, 96, 137, 137])
        features = self.pool1(features)
        # print(features.size())    # torch.Size([batch_size, 96, 68, 68])
        features = self.conv2b(self.conv2a(features)) + self.conv2c(features)
        # print(features.size())    # torch.Size([batch_size, 128, 68, 68])
        features = self.pool2(features)
        # print(features.size())    # torch.Size([batch_size, 128, 34, 34])
        features = self.conv3b(self.conv3a(features)) + self.conv3c(features)
        # print(features.size())    # torch.Size([batch_size, 128, 34, 34])
        features = self.pool3(features)
        # print(features.size())    # torch.Size([batch_size, 256, 17, 17])
        features = self.conv4a(features)
        # print(features.size())    # torch.Size([batch_size, 256, 17, 17])
        features = self.conv4b(features)
        # print(features.size())    # torch.Size([batch_size, 256, 17, 17])
        features = self.pool4(features)
        # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
        features = self.conv5b(self.conv5a(features)) + self.conv5c(features)
        # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
        features = self.pool5(features)
        # print(features.size())    # torch.Size([batch_size, 256, 4, 4])
        features = self.conv6a(features)
        # print(features.size())    # torch.Size([batch_size, 256, 4, 4])
        features = self.conv6b(features)
        # print(features.size())    # torch.Size([batch_size, 256, 4, 4])
        features = self.pool6(features)
        # print(features.size())    # torch.Size([batch_size, 256, 2, 2])
        features = self.fc7(features.view(-1, 1024))
        # print(features.size())    # torch.Size([batch_size, 8192])

        gen_voxel = features.view(-1, 128, 4, 4, 4)
        # print(gen_voxel.size())   # torch.Size([batch_size, 128, 4, 4, 4])
        gen_voxel = self.unpool1b(self.unpool1a(gen_voxel)) + self.unpool1c(gen_voxel)
        # print(gen_voxel.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        gen_voxel = self.unpool2b(self.unpool2a(gen_voxel)) + self.unpool2c(gen_voxel)
        # print(gen_voxel.size())   # torch.Size([batch_size, 128, 16, 16, 16])
        gen_voxel = self.unpool3b(self.unpool3a(gen_voxel)) + self.unpool3c(gen_voxel)
        # print(gen_voxel.size())   # torch.Size([batch_size, 64, 32, 32, 32])
        gen_voxel = self.unpool4b(self.unpool4a(gen_voxel)) + self.unpool4c(gen_voxel)
        # print(gen_voxel.size())   # torch.Size([batch_size, 32, 32, 32, 32])
        gen_voxel = self.unpool5b(self.unpool5a(gen_voxel)) + self.unpool5c(gen_voxel)
        # print(gen_voxel.size())   # torch.Size([batch_size, 8, 32, 32, 32])
        gen_voxel = self.unpool6(gen_voxel)
        # print(gen_voxel.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        return gen_voxel
