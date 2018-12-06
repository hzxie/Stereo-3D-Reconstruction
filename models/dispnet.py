# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class DispNet(torch.nn.Module):
    def __init__(self, cfg):
        super(DispNet, self).__init__()
        self.cfg = cfg

        # Encoder
        self.conv1a = torch.nn.Sequential(
            torch.nn.Conv2d(6, 48, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv1b = torch.nn.Sequential(
            torch.nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv1c = torch.nn.Sequential(
            torch.nn.Conv2d(48, 48, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

        self.conv2a = torch.nn.Sequential(
            torch.nn.Conv2d(48, 96, kernel_size=3, padding=1, stride=2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv2b = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

        self.conv3a = torch.nn.Sequential(
            torch.nn.Conv2d(96, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.conv3b = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

        self.conv4a = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.conv4b = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.conv4c = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, dilation=2, padding=2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )

        # Decoder
        self.unpool1a = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool1b = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool1c = torch.nn.Conv2d(128, 2, kernel_size=3, padding=1)

        self.unpool2a = torch.nn.Sequential(
            torch.nn.Conv2d(258, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool2b = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool2c = torch.nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.unpool2d = torch.nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, bias=False, padding=1)
        self.unpool2e = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

        self.unpool3a = torch.nn.Sequential(
            torch.nn.Conv2d(194, 96, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool3b = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool3c = torch.nn.Conv2d(96, 2, kernel_size=3, padding=1)
        self.unpool3d = torch.nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, bias=False, padding=1)
        self.unpool3e = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

        self.unpool4a = torch.nn.Sequential(
            torch.nn.Conv2d(50, 48, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool4b = torch.nn.Sequential(
            torch.nn.Conv2d(48, 48, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool4c = torch.nn.Conv2d(48, 2, kernel_size=3, padding=1)
        self.unpool4d = torch.nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, bias=False, padding=1)
        self.unpool4e = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

        self.unpool5a = torch.nn.Sequential(
            torch.nn.Conv2d(34, 32, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool5b = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.unpool5c = torch.nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def forward(self, left_rgb_images, right_rgb_images):
        rgb_images = torch.cat((left_rgb_images, right_rgb_images), dim=1)
        # print(rgb_images.size())      # torch.Size([batch_size, 6, 137, 137])

        img_features1 = self.conv1c(self.conv1b(self.conv1a(rgb_images)))
        # print(img_features1.size())     # torch.Size([batch_size, 48, 69, 69])
        img_features2 = self.conv2b(self.conv2a(img_features1))
        # print(img_features2.size())     # torch.Size([batch_size, 96, 35, 35])
        img_features3 = self.conv3b(self.conv3a(img_features2))
        # print(img_features3.size())     # torch.Size([batch_size, 128, 18, 18])
        img_features4 = img_features3 + self.conv4a(img_features3)
        # print(img_features4.size())     # torch.Size([batch_size, 128, 18, 18])
        img_features4 = img_features4 + self.conv4b(img_features4)
        # print(img_features4.size())     # torch.Size([batch_size, 128, 18, 18])
        img_features4 = img_features4 + self.conv4c(img_features4)
        # print(img_features4.size())     # torch.Size([batch_size, 128, 18, 18])

        disp_features1a = self.unpool1a(img_features4)
        # print(disp_features1a.size())   # torch.Size([batch_size, 128, 18, 18])
        disp_features1b = self.unpool1b(disp_features1a)
        # print(disp_features1b.size())   # torch.Size([batch_size, 128, 18, 18])
        disp_features1c = self.unpool1c(disp_features1a)
        # print(disp_features1c.size())   # torch.Size([batch_size, 2, 18, 18])
        disp_features1 = torch.cat((img_features3, disp_features1b, disp_features1c), dim=1)
        # print(disp_features1.size())    # torch.Size([batch_size, 258, 18, 18])

        disp_features2b = self.unpool2b(self.unpool2a(disp_features1))
        # print(disp_features2b.size())     # torch.Size([batch_size, 128, 18, 18])
        disp_features2c = disp_features1c + self.unpool2c(disp_features2b)
        # print(disp_features2c.size())     # torch.Size([batch_size, 2, 18, 18])
        disp_features2d = self.unpool2d(disp_features2c)
        # print(disp_features2d.size())     # torch.Size([batch_size, 2, 35, 35])
        disp_features2e = self.unpool2e(disp_features2b)
        # print(disp_features2e.size())     # torch.Size([batch_size, 96, 35, 35])
        disp_features2 = torch.cat((img_features2, disp_features2d, disp_features2e), dim=1)
        # print(disp_features2.size())      # torch.Size([batch_size, 194, 35, 35])

        disp_features3b = self.unpool3b(self.unpool3a(disp_features2))
        # print(disp_features3b.size())     # torch.Size([batch_size, 96, 35, 35])
        disp_features3c = disp_features2d + self.unpool3c(disp_features3b)
        # print(disp_features3c.size())     # torch.Size([batch_size, 2, 35, 35])
        disp_features3d = self.unpool3d(disp_features3c)
        # print(disp_features3d.size())     # torch.Size([batch_size, 2, 69, 69])
        disp_features3e = self.unpool3e(disp_features3b)
        # print(disp_features3e.size())     # torch.Size([batch_size, 48, 69, 69])
        disp_features3 = torch.cat((disp_features3d, disp_features3e), dim=1)
        # print(disp_features3.size())      # torch.Size([batch_size, 50, 69, 69])

        disp_features4b = self.unpool4b(self.unpool4a(disp_features3))
        # print(disp_features4b.size())     # torch.Size([batch_size, 48, 69, 69])
        disp_features4c = disp_features3d + self.unpool4c(disp_features4b)
        # print(disp_features4c.size())     # torch.Size([batch_size, 2, 69, 69])
        disp_features4d = self.unpool4d(disp_features4c)
        # print(disp_features4d.size())     # torch.Size([batch_size, 2, 137, 137])
        disp_features4e = self.unpool4e(disp_features4b)
        # print(disp_features4e.size())     # torch.Size([batch_size, 32, 137, 137])
        disp_features4 = torch.cat((disp_features4d, disp_features4e), dim=1)
        # print(disp_features4.size())      # torch.Size([batch_size, 34, 137, 137])

        disp_features5b = self.unpool5b(self.unpool5a(disp_features4))
        # print(disp_features5b.size())     # torch.Size([batch_size, 32, 137, 137])
        disp_features5c = disp_features4d + self.unpool5c(disp_features5b)
        # print(disp_features5c.size())     # torch.Size([batch_size, 2, 137, 137])

        disparity_maps = torch.split(disp_features5c.clamp(min=0), 1, dim=1)
        # print(disparity_maps[0].size())     # torch.Size([batch_size, 1, 137, 137])
        # print(disparity_maps[1].size())     # torch.Size([batch_size, 1, 137, 137])

        return disparity_maps[0], disparity_maps[1]