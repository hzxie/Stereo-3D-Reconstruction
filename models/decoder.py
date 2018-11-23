# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.unpool1a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool1b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool1c = torch.nn.ConvTranspose3d(128, 128, kernel_size=1, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, output_padding=1)


        self.unpool2a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool2b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool2c = torch.nn.ConvTranspose3d(128, 128, kernel_size=1, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, output_padding=1)


        self.unpool3a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool3b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool3c = torch.nn.ConvTranspose3d(128, 64, kernel_size=1, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, output_padding=1)


        self.unpool4a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool4b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 32, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool4c = torch.nn.ConvTranspose3d(64, 32, kernel_size=1, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS)


        self.unpool5a = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool5b = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 8, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool5c = torch.nn.ConvTranspose3d(32, 8, kernel_size=1, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS)


        self.unpool6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )


    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_voxels = []
        raw_features = []

        for features in image_features:
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
            raw_feature = gen_voxel
            # print(gen_voxel.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            gen_voxel = self.unpool6(gen_voxel)
            raw_feature = torch.cat((raw_feature, gen_voxel), dim=1)
            # print(gen_voxel.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            
            gen_voxels.append(torch.squeeze(gen_voxel, dim=1))
            raw_features.append(raw_feature)

        gen_voxels = torch.stack(gen_voxels).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_voxels.size())        # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())      # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_voxels
