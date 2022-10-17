# -*- coding: utf-8 -*-
# date: 2022/3/24
# Project: HSI-Classification
# File Name: FeaturePyramid.py
# Description: 特征金字塔
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch import nn
import torch.nn.functional as F


class FeaturePyramid(nn.Module):
    def __init__(self, opt, fp_name, **kwargs):
        '''
        选择要使用的特征金字塔处理方式
        :param fp_name: 处理方式名
        '''
        super(FeaturePyramid, self).__init__()
        self.opt = opt
        self.fp_name = fp_name
        if fp_name == 'SPP': self.FPNet = SPP()
        elif fp_name == 'PPM': self.FPNet = PPM()
        elif fp_name == 'ASPP': self.FPNet = ASPP()
        elif fp_name == 'FPN': self.FPNet = FPN()
        elif fp_name == 'PANet': self.FPNet = PANet()

    def forward(self, features):
        '''
        :param features: swin-transformer中获取的特征
             if fp_name in ['FPN', 'PANet']:
               features=[{'x':x0, 'H':H0, 'W':W0},
                        {'x':x1, 'H':H1, 'W':W1},
                        {'x':x2, 'H':H2, 'W':W2},
                        {'x':x3, 'H':H3, 'W':W3},]

             if fp_name in ['SPP', 'PPM', 'ASPP']:
                features={'x':x0, 'H':H0, 'W':W0}
        :return: 返回进行融合后的特征
        '''
        x = self.FPNet(features)

        return x


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, features):
        B, x, H, W, C = features['x'].shape[0], features['x'], features['H'], features['W'], features['x'].shape[2]
        change_H_W = (H % 2 != 0) and (W % 2 != 0)  # H和W 都为 奇数时 需要调整尺寸
        if change_H_W:
            # 调整尺寸
            change_H, change_W = H + 1, W + 1
        else:
            # 此时不用调整
            change_H, change_W = H, W   # 4, 4

        x = x.view(-1, C, H*W)   # [b, C, H*W]  [b, 96, 4*4]
        up_linear = nn.Linear(in_features=H*W, out_features=change_H*change_W*16*16).cuda()
        x = up_linear(x).view(-1, C, change_H*16, change_W*16)  # [b, 96, 64*64] -> [b, 96, 64, 64]
        x = nn.ReLU()(x)
        # x = F.interpolate(x, size=(change_H*16, change_W*16), mode='nearest')   # [b, 96, 64, 64]
        spp_list = []
        for i in range(4):
            new_H, new_W = change_H*pow(2, i), change_W*pow(2, i)     # 4, 4
            maxpool = nn.MaxPool2d(kernel_size=(new_H, new_W), stride=(new_H, new_W)).cuda()
            spp = maxpool(x).view(B, C, -1)   # [b, 96, 16*16]
            spp_list.append(spp)

        x = torch.cat(spp_list, dim=2)  # 在最后一个维度上进行合并  [b, 96, 16*16+8*8+4*4+2*2]
        linear = nn.Linear(in_features=x.shape[-1], out_features=H*W).cuda()  # 使用1x1卷积对通道进行变换
        x = linear(x).permute(0, 2, 1)  # [b, 96, 4*4] -> [b, 4*4, 96]
        x = nn.ReLU()(x)
        return x


class PPM(nn.Module):
    def __init__(self):
        super(PPM, self).__init__()

    def forward(self, features):
        B, x, H, W, C = features['x'].shape[0], features['x'], features['H'], features['W'], features['x'].shape[2]
        change_H_W = (H % 2 != 0) and (W % 2 != 0)  # H和W 都为 奇数时 需要调整尺寸
        if change_H_W:
            # 调整尺寸
            change_H, change_W = H + 1, W + 1
        else:
            # 此时不用调整
            change_H, change_W = H, W  # 4, 4

        x = x.view(-1, C, H, W)
        ppm_list = []
        for i in range(1, 5):   # 生成4个maxpool层
            new_H, new_W = change_H*pow(2, i), change_W*pow(2, i)   # 8, 8
            x = F.interpolate(x, size=(new_H, new_W), mode='nearest')   # [b, 96, 8, 8]
            maxpool = nn.MaxPool2d(kernel_size=(new_H//4, new_W//4), stride=(new_H//4)).cuda()     # 因为 每次都是偶数进行变换，因此不用进行padding
            ppm = maxpool(x)  # [b, C, H, W] [b, 96, 4, 4]
            ppm_list.append(ppm)

        x = torch.cat(ppm_list, dim=1)  # 在通道维 上 进行合并  [b, C*4, H, W] [b, 96*4, 4, 4]
        conv1x1 = nn.Conv2d(in_channels=C*4, out_channels=C, kernel_size=1).cuda()   # 使用1x1卷积对通道进行变换
        x = conv1x1(x)  # [b, 96, 4, 4]
        x = nn.ReLU()(x)
        x = F.interpolate(x, size=(H, W), mode='nearest').view(-1, H*W, C)   # 将数据还原为输入前的shape

        return x


class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

    def forward(self, features):
        B, x, H, W, C = features['x'].shape[0], features['x'], features['H'], features['W'], features['x'].shape[2]
        change_H_W = (H % 2 != 0) and (W % 2 != 0)  # H和W 都为 奇数时 需要调整尺寸
        if change_H_W:
            # 调整尺寸
            change_H, change_W = H + 1, W + 1
        else:
            # 此时不用调整
            change_H, change_W = H, W  # 4, 4

        x = x.view(-1, C, H * W)  # [b, C, H*W]  [b, 96, 4*4]
        up_linear = nn.Linear(in_features=H * W, out_features=change_H * change_W * 16 * 16).cuda()
        x = up_linear(x).view(-1, C, change_H * 16, change_W * 16)  # [b, 96, 64*64] -> [b, 96, 64, 64]
        x = nn.ReLU()(x)
        temp_channel = 128
        # ======> 1
        aspp = nn.Conv2d(in_channels=C, out_channels=temp_channel, kernel_size=1, bias=False).cuda()(x)
        aspp = nn.BatchNorm2d(temp_channel).cuda()(aspp)
        aspp= nn.ReLU().cuda()(aspp)
        # ======> 2
        aspp_list = [aspp]
        dilation_list = [2, 4, 6]  # 深度可分离卷积 内核元素之间的间距
        for i, dilation in enumerate(dilation_list):
            aspp = nn.Conv2d(in_channels=C, out_channels=temp_channel, kernel_size=3, padding=dilation, dilation=dilation, bias=False).cuda()(x)
            aspp = nn.BatchNorm2d(temp_channel).cuda()(aspp)
            aspp = nn.ReLU().cuda()(aspp)
            aspp_list.append(aspp)
        # ======> 3
        aspp = nn.AdaptiveAvgPool2d(1).cuda()(x)
        aspp = nn.Conv2d(in_channels=C, out_channels=temp_channel, kernel_size=1, bias=False).cuda()(aspp)
        aspp = nn.BatchNorm2d(temp_channel).cuda()(aspp)
        aspp = nn.ReLU().cuda()(aspp)
        aspp = F.interpolate(aspp, size=(change_H * 16, change_W * 16), mode='bilinear', align_corners=False)
        aspp_list.append(aspp)
        # ======> 4
        x = torch.cat(aspp_list, dim=1)
        x = nn.Conv2d(in_channels=temp_channel*5, out_channels=temp_channel, kernel_size=1, bias=False).cuda()(x)
        x = nn.BatchNorm2d(temp_channel).cuda()(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.5).cuda()(x)
        # ======> 5
        x = nn.Conv2d(in_channels=temp_channel, out_channels=C, kernel_size=1, bias=False).cuda()(x)
        x = F.interpolate(x, size=(H, W), mode='nearest').view(-1, H*W, C)

        return x



class FPN(nn.Module):
    '''FPN是一个有高到低 的特征融合的过程'''
    def __init__(self):
        super(FPN, self).__init__()
        # 对C5减少通道数，得到P5
        self.toplayer = nn.Linear(768, 192)
        # 横向连接保证通道数相同
        self.latlayer1 = nn.Linear(768, 192)  # C4
        self.latlayer2 = nn.Linear(384, 192)  # C3
        self.latlayer3 = nn.Linear(192, 192)  # C2
        # 卷积融合，平滑处理
        self.smooth = nn.Linear(192, 768)

    def _upsample_add(self, P, C, H, W):
        # P表示高一层特征
        # C表示低一层特征 H W表示低一层特征图的高宽
        P = P.transpose(1, 2)   # [b, C, H*W]
        P = F.interpolate(P, size=H*W)   # 修改特征图尺寸，使前后可以相加
        return P.transpose(1, 2) + C

    def forward(self, features):
        # C: x[b, H*W, C], H, W
        # 层次由高到低依次为 C5(768) -> C4(768) -> C3(384) -> C2(192)
        # 自下而上获取所有特征层的信息
        C2, C3, C4, C5 = features[0], features[1], features[2], features[3]
        # 自上而下进行特征融合
        P5 = self.toplayer(C5['x'])     # [b, H*W, 192]
        P4 = self._upsample_add(P5, self.latlayer1(C4['x']), C4['H'], C4['W'])  # [b, H*W, 192]
        P3 = self._upsample_add(P4, self.latlayer2(C3['x']), C3['H'], C3['W'])  # [b, H*W, 192]
        P2 = self._upsample_add(P3, self.latlayer3(C2['x']), C2['H'], C2['W'])  # [b, H*W, 192]
        # 卷积融合，平滑处理
        x = self.smooth(P2)  # [b, H*W, 768]
        return x


class PANet(nn.Module):
    '''PANet 是一个由 低到高 的特征融合的过程'''
    def __init__(self):
        super(PANet, self).__init__()
        # 构建 3组 q
        self.proj_q1 = nn.Linear(384, 384)
        self.proj_q2 = nn.Linear(768, 768)
        self.proj_q3 = nn.Linear(768, 768)
        # 构建 3组 k
        self.proj_k1 = nn.Linear(384, 384)
        self.proj_k2 = nn.Linear(768, 768)
        self.proj_k3 = nn.Linear(768, 768)
        # 构建 3组 v
        self.proj_v1 = nn.Linear(192, 384)
        self.proj_v2 = nn.Linear(384, 768)
        self.proj_v3 = nn.Linear(768, 768)

        self.softmax = nn.Softmax(dim=-1)

    def _make_attention(self, q, k):
        '''计算SA Attention'''
        # q:[b, H*W, C]  k:[b, H*W, C]
        attention = torch.bmm(q.transpose(1, 2), k)
        return self.softmax(attention)

    def _downsample_add(self, P, C, H, W):
        # P表示低一层特征
        # C表示高一层特征 H W表示高一层特征图的高宽
        P = P.transpose(1, 2)   # [b, C, H*W]
        P = F.interpolate(P, size=H*W)   # 修改特征图尺寸，使前后可以相加
        return P.transpose(1, 2) + C

    def forward(self, features):
        # C: x[b, H*W, C], H, W
        # 处理层次 由低到高依次为  C2(192) -> C3(384) -> C4(768) -> C5(768)
        # 自下而上获取所有特征层的信息
        C2, C3, C4, C5 = features[0], features[1], features[2], features[3]
        # 第一层融合
        x = self._downsample_add(self.proj_v1(C2['x']), C3['x'], C3['H'], C4['W'])  # [b, H*W, 384]
        q1, k1 = self.proj_q1(C3['x']), self.proj_k1(C3['x'])  # [b, H*W, 384]  [b, H*W, 384]
        attention = self._make_attention(q1, k1)  # [b, 384, 384]
        x = torch.bmm(x, attention)  # [b, H*W, 384]
        # 第二层融合
        x = self._downsample_add(self.proj_v2(x), C4['x'], C4['H'], C4['W'])  # [b, H*W, 768]
        q2, k2 = self.proj_q2(C4['x']), self.proj_k2(C4['x'])  # [b, H*W, 768]  [b, H*W, 768]
        attention = self._make_attention(q2, k2)  # [b, 768, 768]
        x = torch.bmm(x, attention)  # [b, H*W, 768]
        # 第三层融合
        x = self._downsample_add(self.proj_v3(x), C5['x'], C5['H'], C5['W'])  # [b, H*W, 768]
        q3, k3 = self.proj_q3(C5['x']), self.proj_k3(C5['x'])  # [b, H*W, 768]  [b, H*W, 768]
        attention = self._make_attention(q3, k3)  # [b, 768, 768]
        x = torch.bmm(x, attention)  # [b, H*W, 768]

        return x

if __name__ == '__main__':
    FeaturePyramid(1, 'SPP')