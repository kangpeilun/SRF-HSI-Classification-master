# -*- coding: utf-8 -*-
# date: 2022/3/24
# Project: HSI-Classification
# File Name: loss.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable


class focal_loss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        # 代码会自动确定类别数
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha,1-alpha])  # 如果 alpha是一个数，此时进行的是二分类
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)    # 如果 alpha是一个list，进行的是多分类
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            # 如果aplha不是None，需要乘上 alpha
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


if __name__ == '__main__':
    pass