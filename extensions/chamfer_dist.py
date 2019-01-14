# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://discuss.pytorch.org/t/fastest-way-to-find-nearest-neighbor-for-a-set-of-points/5938/12
# - https://github.com/ThibaultGROUEIX/AtlasNet/blob/master/training/train_AE_AtlasNet.py#L41-L65

import torch

import utils.network_utils


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def _batch_pairwise_dist(self, x, y):
        bs, n_points, _ = x.size()

        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        xy = torch.bmm(x, y.transpose(2, 1))

        diag_ind = torch.arange(0, n_points).type(torch.LongTensor)
        diag_ind = utils.network_utils.var_or_cuda(diag_ind)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)

        return rx.transpose(2, 1) + ry - 2 * xy

    def forward(self, x, y):
        dist = self._batch_pairwise_dist(x, y)
        dist1, _ = dist.min(dim=1)
        dist2, _ = dist.min(dim=2)

        return torch.mean(dist1) + torch.mean(dist2)
