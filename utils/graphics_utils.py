#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

#构造并返回一个 4x4 的 world→view 变换矩阵（Rt）
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

#CG之视图变换
#考虑相机中心变换的情况下，构造并返回一个 4x4 的 world→view 变换矩阵（Rt）
#先将相机中心在世界坐标下平移并缩放后再反算回新的 world→view，使相机中心受 translate/scale 影响。
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)  # 把 Rt 视作 world→view 的逆（得到 camera→world）
    cam_center = C2W[:3, 3]  # 提取相机中心（world 坐标）
    cam_center = (cam_center + translate) * scale  # 对相机中心应用平移与缩放
    C2W[:3, 3] = cam_center  # 把修改后的相机中心写回
    Rt = np.linalg.inv(C2W)  # 重新求逆得到新的 world→view
    return np.float32(Rt)

#CG之投影变换
#构造并返回一个 4x4 的 projection 矩阵（P）
#用于将视图空间点投影到裁剪空间（NDC），基于给定近/远裁剪面和水平/垂直视野角。
#这里是为了线性化方便反向传播 以及保留深度信息 确定渲染顺序
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))