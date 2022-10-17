# -*- coding: utf-8 -*-
# date: 2022/3/20 21:13
# Project: Swin-Transformer-New
# File Name: HSISwinNet.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import numpy as np
import torch
from torch import nn

from .SwinTransformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224
from .Attention import Pixel_Attention, Band_Attention, Adaptive_Spectral_Spatial_Attention
from utils.dataset import gain_band_patch


def load_pre_trained_model(model, model_type='tiny', device='cuda:0'):
    '''加载模型预训练权重'''
    pretrained_model_path = r'model_data/swin_{}_patch4_window7_224.pth'.format(model_type)   # 获取权重存放路径

    param = torch.load(pretrained_model_path, map_location=torch.device(device))
    # param = torch.load(pretrained_model_path, map_location='cuda:0')  # 加载预训练权重
    model_dict = model.state_dict()  # 获取原模型初始化权重参数
    state_dict = {k: v for idx, (k, v) in enumerate(param.items()) if idx != 0}  # 修改要加载的参数。不加载 patch_embed.proj.weight(第0层) 层的参数，因为通道数可能真之前的不一致
    model_dict.update(state_dict)  # 将预训练权重替换原来的模型权重
    model.load_state_dict(model_dict)


class MyNet(nn.Module):
    def __init__(self, opt, band, in_chans, num_classes, use_pretrained=True):
        super(MyNet, self).__init__()
        self.band = band
        self.opt = opt

        if opt.use_attention:   # 加载注意力模块
            self.ASSA = Adaptive_Spectral_Spatial_Attention(band, reduction=2)
            new_in_chans = (band - 7) // 2 + 1
            if not opt.use_fusion: # 不使用 特征融合
                self.change_channel = nn.Conv2d(in_channels=24 * new_in_chans, out_channels=band, kernel_size=(1, 1), bias=False)
                self.swin_backbone = swin_tiny_patch4_window7_224(opt=opt, in_chans=band, num_classes=num_classes)
            else:  # 使用特征融合
                self.change_channel = nn.Conv2d(in_channels=24 * new_in_chans, out_channels=band, kernel_size=(1, 1), bias=False)
                self.swin_backbone = swin_tiny_patch4_window7_224(opt=opt, in_chans=in_chans, num_classes=num_classes)
        else:
            self.swin_backbone = swin_tiny_patch4_window7_224(opt=opt, in_chans=in_chans, num_classes=num_classes)

        if use_pretrained:  # 记载预训练权重
            load_pre_trained_model(self.swin_backbone, model_type=opt.swin_type, device=opt.device)


    def format_data(self, data, data_info):
        '''修改band_pixel_patch mode下attention后的特征尺寸'''
        batch_size = data.shape[0]
        feature_map_size = data_info['feature_map_size'].numpy()[0]
        mirror_hsi = data.cpu().detach().numpy()   # [64, 9, 9, 200]
        final_mirror_hsi = np.zeros((batch_size, self.opt.band_patch * pow(self.opt.pixel_patch, 2), feature_map_size,
                                     feature_map_size))  # 存放最终输出的数据，通道 是由 band_patch和pixel_patch共同决定的

        for batch_id in range(batch_size):
            index_i = 0  # 用于记录当前是哪一个 pixel
            index_j = 1  # 用于记录当前是哪一个 pixel
            for x in range(self.opt.pixel_patch):
                for y in range(self.opt.pixel_patch):
                    mirror_hsi_zeros = np.zeros(
                        (1, 1, feature_map_size * feature_map_size))  # 将一维band映射到二维 feature_map中，同时在末尾补0
                    temp_zeros = np.zeros(
                        (1, 1, feature_map_size * feature_map_size))  # temp_zeros用于patch时暂时存放变形后的mirror_hsi数据
                    mirror_hsi_zeros[:, :, :self.band] = mirror_hsi[batch_id, x, y, :]  # 找到原数据对应索引位置处的data cube
                    temp_mirror_hsi = mirror_hsi_zeros  # temp_mirror_hsi用于记录通道仍为1维状态的数值，方便之后进行patch
                    mirror_hsi_zeros = mirror_hsi_zeros.reshape((1, feature_map_size, feature_map_size))
                    # ================进行镜像操作==========================
                    mirror_hsi_zeros = gain_band_patch(mirror_hsi_zeros, temp_mirror_hsi, temp_zeros, self.band,
                                                       feature_map_size,
                                                       self.opt.band_patch) if self.opt.band_patch > 1 else mirror_hsi_zeros  # [band_patch, feature_map_size, feature_map_size]
                    final_mirror_hsi[batch_id, index_i * self.opt.band_patch:index_j * self.opt.band_patch, :, :] = mirror_hsi_zeros
                    index_i += 1  # 索引加一
                    index_j += 1  # 索引加一

        data = torch.from_numpy(final_mirror_hsi).type(torch.FloatTensor).to(self.opt.device)
        return data


    def forward(self, x, data_info):
        '''
        if mode == 'band_patch':
            x:[b, band_patch, feature_map, feature_map]
            data_info: 'band_ori_data':[b, 1, 1, 200]
        if mode == 'pixel_patch':
            x:[b, band, pixel_patch, pixel_patch]
            data_info: 'pixel_ori_data':[b, pixel_patch, pixel_patch, band]
        if mode == 'band_pixel_patch':
            x:[b, band_patch*pow(pixel_patch,2), pixel_patch, pixel_patch]
            data_info: 'band_ori_data':[b, 1, 1, 200]
                       'pixel_ori_data':[b, pixel_patch, pixel_patch, band]
        '''
        if self.opt.use_attention:
            if self.opt.mode == 'pixel_patch':
                x_assa = self.ASSA(data_info['pixel_ori_data'].to(self.opt.device))   # [b, 24*97, h, w]
                ori_x = x   # 保存原始输入的x
                x = x_assa
                if self.opt.change_channel:
                    x = self.change_channel(x_assa)  # [b, c, w, h]
                if self.opt.use_fusion:
                    change_channel = self.change_channel(x_assa)
                    x = ori_x + change_channel  # 原始输入，和attention处理后的特征融合

            elif self.opt.mode == 'band_pixel_patch':
                x_assa = self.ASSA(data_info['pixel_ori_data'].to(self.opt.device))   # [b, 24*97, h, w]
                ori_x = x
                x = x_assa
                if self.opt.change_channel:     # 不进行特征融合，但改变通道
                    x = self.change_channel(x_assa)  # [b, c, w, h]
                if self.opt.use_fusion:
                    change_channel = self.change_channel(x_assa).permute(0, 2, 3, 1)  # [b, c, w, h]
                    change_channel = self.format_data(change_channel, data_info)
                    x = ori_x + change_channel

        x = self.swin_backbone(x)

        return x


if __name__ == '__main__':
    model = MyNet(1, 3, 16)
    # model = swin_tiny_patch4_window7_224(in_chans=1, num_classes=3)
    print(model)