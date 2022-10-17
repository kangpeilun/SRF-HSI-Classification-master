# -*- coding: utf-8 -*-
# date: 2022/9/22
# Project: HSI-Classification
# File Name: vision_of_data.py
# Description: 可视化不同mat数据
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from scipy import io
import numpy as np
from PIL import Image


def read_mat_data(path):
    mat_data = io.loadmat(path)
    input_data = mat_data['input']
    print(input_data.shape)
    print(type(input_data))
    img_array = input_data[:, :, :3]    # 对哪几个通道的数据进行可视化

    ndarray_to_img(path, img_array)


def ndarray_to_img(path, array):
    '''将ndarray转换为PIL 最后保存为图像'''
    img = Image.fromarray(np.uint8(array))
    img.save(f'{path.replace("mat", "png")}')


if __name__ == '__main__':
    mat_data_list = ["./IndianPine.mat",
                     "./Pavia.mat",
                     "./Houston.mat"]
    for mat_data in mat_data_list:
        read_mat_data(mat_data)
        # break