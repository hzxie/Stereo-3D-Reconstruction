# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        self.unpool1a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, bias=False, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool1b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool1c = torch.nn.ConvTranspose3d(256, 128, kernel_size=1, stride=2, bias=False, output_padding=1)

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

        self.unpool5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=3, stride=1, bias=False, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, left_img_features, right_img_features):
        left_img_features = left_img_features.view(-1, 128, 4, 4, 4)
        right_img_features = right_img_features.view(-1, 128, 4, 4, 4)

        gen_voxel = torch.cat((left_img_features, right_img_features), dim=1)
        # print(gen_voxel.size())     # torch.Size([batch_size, 256, 4, 4, 4])
        gen_voxel = self.unpool1b(self.unpool1a(gen_voxel)) + self.unpool1c(gen_voxel)
        # print(gen_voxel.size())     # torch.Size([batch_size, 128, 8, 8, 8])
        gen_voxel = self.unpool2b(self.unpool2a(gen_voxel)) + self.unpool2c(gen_voxel)
        # print(gen_voxel.size())     # torch.Size([batch_size, 128, 16, 16, 16])
        gen_voxel = self.unpool3b(self.unpool3a(gen_voxel)) + self.unpool3c(gen_voxel)
        # print(gen_voxel.size())     # torch.Size([batch_size, 64, 32, 32, 32])
        gen_voxel = self.unpool4b(self.unpool4a(gen_voxel)) + self.unpool4c(gen_voxel)
        # print(gen_voxel.size())     # torch.Size([batch_size, 32, 32, 32, 32])
        gen_voxel = self.unpool5(gen_voxel)
        # print(gen_voxel.size())     # torch.Size([batch_size, 1, 32, 32, 32])
        return gen_voxel.squeeze(dim=1)