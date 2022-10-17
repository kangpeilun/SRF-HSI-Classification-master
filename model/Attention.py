# -*- coding: utf-8 -*-
# date: 2022/4/13 14:44
# Project: HSI-Classification
# File Name: attention.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch import nn

import numpy as np


class Pixel_Attention(nn.Module):
    '''像素 注意力机制, 获取分数'''
    def __init__(self, band):
        super(Pixel_Attention, self).__init__()
        self.q = nn.Conv2d(in_channels=band, out_channels=band//2, kernel_size=(1, 1), bias=False)
        self.k = nn.Conv2d(in_channels=band, out_channels=band//2, kernel_size=(1, 1), bias=False)
        self.v = nn.Conv2d(in_channels=band, out_channels=band, kernel_size=(1, 1), bias=False)
        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [b, c, w, h]
        c, w, h = x.size(1), x.size(2), x.size(3)
        q_mat = self.q(x).view(-1, c, w*h)
        k_mat = self.k(x).view(-1, c, w*h)
        v_mat = self.v(x).view(-1, c, w*h)
        attention = torch.bmm(q_mat, k_mat.transpose(2, 1))
        attention = self.softmax(attention)     # 获取注意力分数  [b, w*h, w*h]
        PA_x = torch.bmm(v_mat, attention).view(-1, c, w, h)  # [b, c, w, h]
        return PA_x


class Band_Attention(nn.Module):
    '''光谱 注意力机制, 获取分数'''
    def __init__(self, band):
        super(Band_Attention, self).__init__()
        self.q = nn.Linear(in_features=band, out_features=band//2, bias=False)
        self.k = nn.Linear(in_features=band, out_features=band//2, bias=False)
        self.v = nn.Linear(in_features=band, out_features=band//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [b, c, 1, 1]
        c, w, h = x.size(1), x.size(2), x.size(3)
        x = x.permute(0, 2, 3, 1)
        q_mat = self.q(x)   # [b, 1, 1, c]
        k_mat = self.k(x)
        v_mat = self.v(x)
        attention = q_mat * k_mat   # [b, 1, 1, c]
        attention = self.sigmoid(attention)
        BA_x = v_mat * attention
        BA_x = BA_x.permute(0, 3, 1, 2)
        return BA_x


class Adaptive_Spectral_Spatial_Attention(nn.Module):
    def __init__(self, band, reduction=2):
        super(Adaptive_Spectral_Spatial_Attention, self).__init__()
        self.conv1x1 = nn.Conv3d(
            in_channels=1,
            out_channels=24,  # 24
            kernel_size=(1, 1, 7),
            stride=(1, 1, 2),
            padding=0)
        self.conv3x3 = nn.Conv3d(
            in_channels=1,
            out_channels=24,  # 24
            kernel_size=(3, 3, 7),
            stride=(1, 1, 2),
            padding=(1, 1, 0))

        self.batch_norm1x1 = nn.Sequential(
            nn.BatchNorm3d(
                24, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.batch_norm3x3 = nn.Sequential(
            nn.BatchNorm3d(
                24, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(
                24, band // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(
            band // reduction, 24, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x:[b, c, h, w]
        batch, w, h, c = x.size(0), x.size(1), x.size(2), x.size(3)
        # x = x.permute(0, 3, 1, 2).unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        x_1x1 = self.conv1x1(x)  # [b, 24, 9, 9, 97]
        x_1x1 = self.batch_norm1x1(x_1x1).unsqueeze(dim=1)  # [b, 1, 24, 9, 9, 97]
        x_3x3 = self.conv3x3(x)  # [b, 24, 9, 9, 97]
        x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)  # [b, 1, 24, 9, 9, 97]

        x1 = torch.cat([x_3x3, x_1x1], dim=1)  # [b, 2, 24, 9, 9, 97]
        U = torch.sum(x1, dim=1)  # [b, 24, 9, 9, 97]
        S = self.pool(U)  # [b, 24, 1, 1, 1]
        Z = self.conv_se(S)  # [b, 100, 1, 1, 1]
        attention_vector = torch.cat(
            [
                self.conv_ex(Z).unsqueeze(dim=1),
                self.conv_ex(Z).unsqueeze(dim=1)
            ],
            dim=1)  # [b, 2, 24, 9, 9, 97]
        attention_vector = self.softmax(attention_vector)  # [b, 2, 24, 1, 1, 1]
        V = (x1 * attention_vector).sum(dim=1).view(batch, -1, h, w)  # [b, 24, 9, 9, 97] -> [b, 24*97, h, w]

        return V