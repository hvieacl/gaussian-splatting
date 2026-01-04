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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid #相机的唯一标识符
        self.colmap_id = colmap_id #colmap中相机位姿的ID
        self.R = R #投影转换旋转矩阵
        self.T = T #投影转换平移矩阵
        self.FoVx = FoVx #相机的水平视场角
        self.FoVy = FoVy #相机的垂直视场角
        self.image_name = image_name #图像文件名

        # 设置数据设备
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        #把输入 PIL图像转成训练用的张量
        resized_image_rgb = PILtoTorch(image, resolution)
        #取前三通道作为 RGB 真实图像（不含 alpha）
        gt_image = resized_image_rgb[:3, ...]

        #alpha 掩码创建
        self.alpha_mask = None
        #如果输入有 4 个通道（RGBA），用第 4 通道作为 self.alpha_mask 并移动到 self.data_device
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        #否则创建一个全 1 的 alpha_mask（表示完全不透明）
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        #train/test 暂停曝光处理
        #当启用且这是一个“测试视图”时，会把掩码的一半置为 0，以在同一张图上分别保留训练/测试区域。
        if train_test_exp and is_test_view:
            #如果是 is_test_dataset（表示整个图像集合属于测试集），则把掩码的左半边设为 0；否则把右半边设为 0。
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        #把真实图像归一化压缩后移动到 self.data_device，并保存图像宽高
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        #处理逆深度图，将其缩放/清理并转换为用于训练的 PyTorch 张量，同时生成一个像素级的可靠性掩码。
        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        #相机视角的最近点和最远点
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        #计算相机的世界到视图变换矩阵、投影矩阵和完整的投影变换矩阵
        #CG相关变换过程
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
#简化版相机类，只保存必要的相机参数
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

