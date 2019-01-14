# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/Point-Set-Generation-Network/blob/master/train_pic2points.py#L31-L53

import torch


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, x, y):
        bs, n_points, points_dim = x.size()

        dist1 = torch.sqrt(self._pairwise_dist(x, y))
        values1, indices1 = dist1.min(dim=2)
        dist2 = torch.sqrt(self._pairwise_dist(y, x))
        values2, indices2 = dist2.min(dim=2)

        a = torch.div(torch.sum(values1, 1), n_points)
        b = torch.div(torch.sum(values2, 1), n_points)

        return torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)

    def _pairwise_dist(self, x, y):
        bs, n_points, points_dim = x.size()

        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))

        diag_ind = torch.arange(0, n_points).type(torch.cuda.LongTensor)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)

        return (rx.transpose(2, 1) + ry - 2 * zz)
