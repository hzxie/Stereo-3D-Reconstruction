# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume, save_dir, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)


def get_ptcloud_views(ptcloud, save_dir, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure()
    ptcloud = np.transpose(ptcloud, (1, 0))
    x, y, z = ptcloud
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')

    save_path = os.path.join(save_dir, 'ptcloud-%06d.png' % n_itr)
    plt.savefig(save_path, dpi=144)
    plt.close()
    return cv2.imread(save_path)
