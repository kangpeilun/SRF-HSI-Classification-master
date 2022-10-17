# -*- coding: utf-8 -*-
# date: 2022/2/13 10:42
# Project: 毕业设计
# File Name: datainfo.py
# Description: 获取数据集所包含的信息
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import scipy.io as sio
import os
import numpy as np

'''
color_map
[[1.   1.   1.  ]
 [1.   1.   0.5 ]
 [0.   0.   1.  ]
 [1.   0.   0.  ]
 [0.   1.   0.25]
 [1.   0.   1.  ]
 [0.5  0.   1.  ]
 [0.   0.5  1.  ]
 [0.   1.   0.  ]
 [0.5  0.5  0.25]
 [0.5  0.   0.5 ]
 [0.   0.5  1.  ]
 [0.   0.25 0.5 ]
 [0.   0.5  0.25]
 [0.5  0.25 0.  ]
 [0.   1.   0.5 ]
 [1.   1.   0.  ]]
---------- IndianPine ----------
input size of IndianPine data:  (145, 145, 200)
train data size(TR):  (145, 145)
test data size(TE):  (145, 145)
the class of the num in this data:  16
---------- Houston ----------
input size of Houston data:  (349, 1905, 144)
train data size(TR):  (349, 1905)
test data size(TE):  (349, 1905)
the class of the num in this data:  15
---------- Pavia ----------
input size of Pavia data:  (610, 340, 103)
train data size(TR):  (610, 340)
test data size(TE):  (610, 340)
the class of the num in this data:  9
'''

AVIRIS_colormap = './AVIRIS_colormap.mat'
IndianPine = './IndianPine.mat'
Houston = './Houston.mat'
Pavia = './Pavia.mat'

colormap_mat = sio.loadmat(AVIRIS_colormap)
print(colormap_mat['mycolormap'])

print('-'*10,'IndianPine','-'*10)
IndianPine_mat = sio.loadmat(IndianPine)
print(IndianPine_mat)
print('input size of IndianPine data: ',IndianPine_mat['input'].shape)
print('train data size(TR): ', IndianPine_mat['TR'].shape)
print('test data size(TE): ', IndianPine_mat['TE'].shape)
print('the class of the num in this data: ', IndianPine_mat['TR'].max())

print('-'*10,'Houston','-'*10)
Houston_mat = sio.loadmat(Houston)
print('input size of Houston data: ',Houston_mat['input'].shape)
print('train data size(TR): ', Houston_mat['TR'].shape)
print('test data size(TE): ', Houston_mat['TE'].shape)
print('the class of the num in this data: ', Houston_mat['TR'].max())

print('-'*10,'Pavia','-'*10)
Pavia_mat = sio.loadmat(Pavia)
print('input size of Pavia data: ',Pavia_mat['input'].shape)
print('train data size(TR): ', Pavia_mat['TR'].shape)
print('test data size(TE): ', Pavia_mat['TE'].shape)
print('the class of the num in this data: ', Pavia_mat['TR'].max())
