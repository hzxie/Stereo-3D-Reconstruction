# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class FusionNet(torch.nn.Module):
    def __init__(self, cfg):
        super(FusionNet, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.conv1a = torch.nn.Sequential(
            torch.nn.Conv2d(8, 96, kernel_size=7, padding=3),
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
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv4b = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv4c = torch.nn.Conv2d(256, 512, kernel_size=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv5a = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv5b = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv5c = torch.nn.Conv2d(512, 512, kernel_size=1)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, left_rgbd_images, right_rgbd_images):
        rgbd_images = torch.cat((left_rgbd_images, right_rgbd_images), dim=1)
        # print(rgbd_images.size())  # torch.Size([batch_size, 8, 137, 137])
        features = self.conv1a(rgbd_images.view(-1, self.cfg.CONST.IMG_C * 2, self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W))
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
        # print(features.size())    # torch.Size([batch_size, 256, 34, 34])
        features = self.pool3(features)
        # print(features.size())    # torch.Size([batch_size, 256, 17, 17])
        features = self.conv4b(self.conv4a(features)) + self.conv4c(features)
        # print(features.size())    # torch.Size([batch_size, 512, 17, 17])
        features = self.pool4(features)
        # print(features.size())    # torch.Size([batch_size, 512, 8, 8])
        features = self.conv5b(self.conv5a(features)) + self.conv5c(features)
        # print(features.size())    # torch.Size([batch_size, 512, 8, 8])
        features = self.pool5(features)
        # print(features.size())    # torch.Size([batch_size, 512, 4, 4])

        return features.view(-1, 8192)
