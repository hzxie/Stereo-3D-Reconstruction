# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

import utils.network_utils


class CorrelationNet(torch.nn.Module):
    def __init__(self, cfg):
        super(CorrelationNet, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.conv1a = torch.nn.Sequential(
            torch.nn.Conv3d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv1b = torch.nn.Sequential(
            torch.nn.Conv3d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv1c = torch.nn.Conv3d(512, 256, kernel_size=1)

        self.conv2a = torch.nn.Sequential(
            torch.nn.Conv3d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv2b = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv2c = torch.nn.Conv3d(256, 128, kernel_size=1)

        self.conv3a = torch.nn.Sequential(
            torch.nn.Conv3d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv3b = torch.nn.Sequential(
            torch.nn.Conv3d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv3c = torch.nn.Conv3d(128, 64, kernel_size=1)

        self.conv4a = torch.nn.Sequential(
            torch.nn.Conv3d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv4b = torch.nn.Sequential(
            torch.nn.Conv3d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv4c = torch.nn.Conv3d(64, 32, kernel_size=1)

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(5, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.fc7 = torch.nn.Linear(1156, 8192)

    def forward(self, left_ll_features, right_ll_features):
        n, c, h, w = left_ll_features.size()
        # assert left_ll_features.size() == right_ll_features.size()

        disp_channels = self.cfg.DATASET.MAX_DISP_VALUE // 4
        cost_volume = utils.network_utils.var_or_cuda(torch.zeros((n, 2 * c, disp_channels, h, w)))
        # print(cost_volume.size())   # torch.Size([20, 512, 5, 34, 34])
        for i in range(0, disp_channels):
            if i == 0:
                cost_volume[:, :c, i, :, i:] = left_ll_features
                cost_volume[:, c:, i, :, i:] = right_ll_features
            else:
                cost_volume[:, :c, i, :, i:] = left_ll_features[:, :, :, i:]
                cost_volume[:, c:, i, :, i:] = right_ll_features[:, :, :, :-i]

        cost_volume = cost_volume.contiguous()
        corr_features = self.conv1b(self.conv1a(cost_volume)) + self.conv1c(cost_volume)
        # print(corr_features.size()) # torch.Size([20, 256, 5, 34, 34])
        corr_features = self.conv2b(self.conv2a(corr_features)) + self.conv2c(corr_features)
        # print(corr_features.size()) # torch.Size([20, 128, 5, 34, 34])
        corr_features = self.conv3b(self.conv3a(corr_features)) + self.conv3c(corr_features)
        # print(corr_features.size()) # torch.Size([20, 64, 5, 34, 34])
        corr_features = self.conv4b(self.conv4a(corr_features)) + self.conv4c(corr_features)
        # print(corr_features.size()) # torch.Size([20, 32, 5, 34, 34])
        corr_features = self.conv5(corr_features).squeeze(dim=1)
        # print(corr_features.size()) # torch.Size([20, 5, 34, 34])
        corr_features = self.conv6(corr_features)
        # print(corr_features.size()) # torch.Size([20, 1, 34, 34])
        corr_features = self.fc7(corr_features.view(-1, 1156))
        # print(corr_features.size()) # torch.Size([20, 8192])

        return corr_features
