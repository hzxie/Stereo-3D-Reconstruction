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
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 256, kernel_size=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.Conv3d(256, 128, kernel_size=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv2a = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv2b = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv3a = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv3b = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv4a = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv4b = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv5a = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv5b = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 32, kernel_size=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.Conv3d(32, 1, kernel_size=1),
            torch.nn.BatchNorm3d(1)
        )
        
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.fc8 = torch.nn.Linear(1156, 4096)

    def forward(self, left_ll_features, right_ll_features):
        n, c, h, w = left_ll_features.size()
        # assert left_ll_features.size() == right_ll_features.size()

        disp_channels = self.cfg.DATASET.MAX_DISP_VALUE // 4
        cost_volume = utils.network_utils.var_or_cuda(torch.zeros((n, 2 * c, disp_channels, h, w)))
        # print(cost_volume.size())   # torch.Size([20, 512, 6, 34, 34])
        for i in range(0, disp_channels):
            if i == 0:
                cost_volume[:, :c, i, :, i:] = left_ll_features
                cost_volume[:, c:, i, :, i:] = right_ll_features
            else:
                cost_volume[:, :c, i, :, i:] = left_ll_features[:, :, :, i:]
                cost_volume[:, c:, i, :, i:] = right_ll_features[:, :, :, :-i]

        cost_volume = cost_volume.contiguous()
        corr_features = self.conv1(cost_volume)
        # print(corr_features.size()) # torch.Size([20, 128, 6, 34, 34])
        corr_features = self.conv2b(self.conv2a(corr_features)) + corr_features
        # print(corr_features.size()) # torch.Size([20, 128, 6, 34, 34])
        corr_features = self.conv3b(self.conv3a(corr_features)) + corr_features
        # print(corr_features.size()) # torch.Size([20, 128, 6, 34, 34])
        corr_features = self.conv4b(self.conv4a(corr_features)) + corr_features
        # print(corr_features.size()) # torch.Size([20, 128, 6, 34, 34])
        corr_features = self.conv5b(self.conv5a(corr_features)) + corr_features
        # print(corr_features.size()) # torch.Size([20, 128, 6, 34, 34])
        corr_features = self.conv6(corr_features).squeeze(dim=1)
        # print(corr_features.size()) # torch.Size([20, 6, 34, 34])
        corr_features = self.conv7(corr_features)
        # print(corr_features.size()) # torch.Size([20, 1, 34, 34])
        corr_features = self.fc8(corr_features.view(-1, 1156))
        # print(corr_features.size()) # torch.Size([20, 4096])

        return corr_features
