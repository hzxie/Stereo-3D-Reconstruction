# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Fire(torch.nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = torch.nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = torch.nn.ReLU(inplace=True)
        self.expand1x1 = torch.nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = torch.nn.ReLU(inplace=True)
        self.expand3x3 = torch.nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        self.layer1 = Fire(80, 20, 80, 80)
        self.layer2 = torch.nn.Sequential(
            Fire(160, 20, 80, 80),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.layer3 = Fire(160, 40, 160, 160)
        self.layer4 = Fire(320, 40, 160, 160)
        self.layer5 = Fire(320, 60, 240, 240)
        self.layer6 = Fire(480, 60, 240, 240)
        self.layer7 = Fire(480, 60, 320, 320)
        self.layer8 = Fire(640, 60, 320, 320)
        self.layer9 = torch.nn.Sequential(
            torch.nn.Conv2d(640, cfg.NETWORK.N_POINTS, kernel_size=3),
            torch.nn.ReLU(inplace=True)
        )
        self.layer10 = torch.nn.Linear(36, 3)

    def forward(self, left_img_features, right_img_features, corr_features):
        left_img_features = left_img_features.view(-1, 32, 16, 16)
        right_img_features = right_img_features.view(-1, 32, 16, 16)
        corr_features = corr_features.view(-1, 16, 16, 16)

        pts_features = torch.cat((left_img_features, right_img_features, corr_features), dim=1)
        # print(pts_features.size())     # torch.Size([batch_size, 80, 16, 16])
        pts_features = self.layer1(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 160, 16, 16])
        pts_features = self.layer2(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 160, 8, 8])
        pts_features = self.layer3(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 320, 8, 8])
        pts_features = self.layer4(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 320, 8, 8])
        pts_features = self.layer5(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 480, 8, 8])
        pts_features = self.layer6(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 480, 8, 8])
        pts_features = self.layer7(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 640, 8, 8])
        pts_features = self.layer8(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 640, 8, 8])
        pts_features = self.layer9(pts_features)
        # print(pts_features.size())     # torch.Size([batch_size, 1024, 6, 6])
        pts = self.layer10(pts_features.view(-1, self.cfg.NETWORK.N_POINTS, 36))

        return pts
