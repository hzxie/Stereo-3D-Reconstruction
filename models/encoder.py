# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
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

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.conv1a(img.view(-1, self.cfg.CONST.IMG_C, self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W))
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
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 8192])
        return image_features
