# -*- coding: utf-8 -*-
# date: 2022/2/20 14:16
# Project: 毕业设计
# File Name: dataset.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch

import numpy as np
from scipy.io import loadmat, savemat
from random import shuffle


def choose_alpha(number_train, alpha_mode, data_name):
    '''为focal loss 选择对应alpha值'''
    temp_number_train = np.array(number_train).astype(float)
    if alpha_mode == 'auto':
        alpha = list(temp_number_train / temp_number_train.sum())
    elif alpha_mode == 'handle':
        if data_name == 'Indian':       # [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 15, 15, 15] 695
            alpha = [0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
        elif data_name == 'Pavia':      # [548, 540, 392, 524, 265, 532, 375, 514, 231] 3921
            alpha = [0.75, 0.75, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
        elif data_name == 'Houston':    # [198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187] 2832  该数据 的比例比较均衡，不设置alpha
            alpha = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75]
    elif alpha_mode == 'None':
        alpha = None

    return alpha


def get_data(data_name):
    '''加载训练数据'''
    if data_name == 'Indian':
        data = loadmat('data/IndianPine.mat')
    elif data_name == 'Pavia':
        data = loadmat('data/Pavia.mat')
    elif data_name == 'Houston':
        data = loadmat('data/Houston.mat')
    else:
        raise ValueError('Unknow Dataset')

    # AVIRIS_colormap.mat的作用是绘制高光谱图像的假彩色影像，其形状为(17, 3), 共记录了17个类别中3个通道中每个通道的值
    color_mat = loadmat('data/AVIRIS_colormap.mat')
    color_matrix = color_mat['mycolormap']  # (17,3)

    input = data['input']  # 高光谱数据中不同通道的反射值，格式：[height, width, band]
    TR = data['TR']  # 训练集标签，同时包含的训练数据在 mat 文件中的坐标
    TE = data['TE']  # 测试集标签，同时包含的测试数据在 mat 文件中的坐标
    label = TR + TE  # 将训练数据与测试数据合并，获得全部的数据标签
    num_classes = np.max(TR)  # 获取 数据 的总类别数

    return input, TR, TE, label, num_classes, color_matrix


def normalize(input):
    '''将input数据在 光谱band 维上进行标准化，加快训练速度'''
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):  # 获取 光谱维
        input_max = np.max(input[:, :, i])  # 获取每一个通道上 所有像素中，光谱反射值的 最大值
        input_min = np.min(input[:, :, i])  # 获取每一个通道上 所有像素中，光谱反射值的 最小值
        input_mean = np.mean(input[:, :, i]) # 获取每一个通道上 所有像素中，光谱反射值的 平均值
        input_std = np.std(input[:, :, i])
        # input_normalize[:, :, i] = (input[:, :, i]-input_mean) / (input_max-input_min)
        input_normalize[:, :, i] = (input[:, :, i]-input_mean) / input_std      # 标准化的效果更好

    return input_normalize


def chooose_train_and_test_point(num_classes, TR, TE, label):
    '''
    分类 时需要专门进行的处理
    :param train_data: 对应TR
    :param test_data: TE
    :param true_data: label
    :param num_classes: 类别数，如IndianPine.mat数据有16个类别
    :return:
    '''
    number_train = []  # 记录训练集中 每一类有多少个含有正确标签的像素 [50, 50, 50, ...]
    pos_train = {}  # 记录训练集中 每一类含有正确标签的像素 所对应的坐标 {0:[pos1,pos2,...], 1:[pos1, pos2, ...], ...}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(TR == (i + 1))  # 因为有16类，而TR使用了0进行占位，故这里需要将TR的类别+1才能得到 正确的类别，然后找到所有正确类别所对应的像素的 坐标
        number_train.append(each_class.shape[0])  # 得到每一类共有 多少个 样本
        pos_train[i] = each_class  # 使用字典记住 每个类别 对应 含有正确标签的像素 所对应的坐标

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        '''
        np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
        np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
        '''
        # 将所有含有正确分类的像素点对应的 坐标组合，得到所有待训练像素点的位置
        # 因为是在pos_train[0]的基础上进行拼接的，故索引从 1 到 num_class-1
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(TE == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    # --------------------------for true data------------------------------------
    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(label == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    train_pos_label = []
    test_pos_label = []
    true_pos_label = []
        # 将数据格式整理为[(pos1, label1), (pos2, label2), ...] 的形式
    for class_id in pos_train.keys():
        for pos in pos_train[class_id]:
            train_pos_label.append((pos, class_id))
    for class_id in pos_test.keys():
        for pos in pos_test[class_id]:
            test_pos_label.append((pos, class_id))
    for class_id in pos_true.keys():
        for pos in pos_true[class_id]:
            true_pos_label.append((pos, class_id))

    # shuffle(train_pos_label)
    return train_pos_label, test_pos_label, true_pos_label, total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def gain_band_patch(mirror_hsi, temp_mirror_hsi, temp_zeros, band, feature_map_size, patch):
    '''
    生成band_patch数据
    :param mirror_hsi: [1, 15, 15] 拥有真实值
    :param temp_mirror_hsi: [1, 1, 15*15] 拥有真实值
    :param temp_zeros: [1, 1, 15*15] 用0进行填充
    :param band: 光谱带条数
    :param patch: 多少个相邻光谱带为一组
    :return:
    '''
    mirror_hsi = np.repeat(mirror_hsi, repeats=patch, axis=0)  # 在通道方向对mirror_hsi数据复制patch次
    # ==============每一个通道进行一次单独的处理=================
    # 循环的次数为patch-1，因为第一个通道的值不需要变动
    for index in range(1, patch):   # 表示 需新生成 patch-1 个心得data cube
        for idx in range(1, index*10+1):   # 每个data cube的band维都需要 依次错位 patch-1个 patch-2个 ...
            temp_zeros[:, :, band-(index*10+1)+idx] = temp_mirror_hsi[:, :, idx-1]
        temp_zeros[:, :, :band-index*10] = temp_mirror_hsi[:, :, index*10:band]
        mirror_hsi[index, :, :] = temp_zeros.reshape((1, feature_map_size, feature_map_size))

    return mirror_hsi


def gain_band_patch_data(index, pos_label, input_normalize, patch=1):
    '''band_patch
    将训练像素的通道展平为一个特征图，(假设有200个通道)
        当patch=1时 数据shape为(15, 15, 1)，patch=2时 shape为(15, 15, 2)，以此类推
        假设特征图大小为2x2，patch=2，下面时两张堆叠的特征图：
            1 2    —>   2 3     shape: [2, 2, 2]
            3 4         4 1
    '''
    _, _, band = input_normalize.shape
    pos, label = pos_label[index]  # 获取指定索引位置的数据
    # =================准备初始mirror_hsi==================
    height_and_width = np.sqrt(band)
    # 只有当无法正好开放时，才需要+1操作
    feature_map_size = int(height_and_width) + 1 if height_and_width > np.floor(height_and_width) else int(np.floor(height_and_width))  # 将一维band映射到二维 feature_map中
    mirror_hsi = np.zeros((1, 1, feature_map_size * feature_map_size))  # 将一维band映射到二位 feature_map中，同时可以在末尾进行补0操作
    temp_zeros = np.zeros((1, 1, feature_map_size * feature_map_size))  # temp_zeros用于patch时暂时存放变形后的mirror_hsi数据
    mirror_hsi[:, :, :band] = input_normalize[pos[0], pos[1], :]  # 找到原数据对应索引位置处的data cube
    temp_mirror_hsi = mirror_hsi   # temp_mirror_hsi用于记录通道仍为1维状态的数值，方便之后进行patch
    mirror_hsi = mirror_hsi.reshape((1, feature_map_size, feature_map_size))
    # =================进行镜像操作==================
    # 只有当patch=1时才需要进行patch操作，否则使用初始数据即可
    mirror_hsi = gain_band_patch(mirror_hsi, temp_mirror_hsi, temp_zeros, band, feature_map_size, patch) if patch>1 else mirror_hsi

    data = torch.from_numpy(mirror_hsi).type(torch.FloatTensor)

    # 获取numpy格式的数据，方便在attention模块中进行处理
    data_info = {
        'pos': pos,
        'band_ori_data': torch.from_numpy(input_normalize[pos[0], pos[1], :].reshape(1, 1, -1)).type(torch.FloatTensor),  # 获取未经展开的原始数据 [c, w, h]
        'feature_map_size': feature_map_size
    }

    return data, label, data_info


def pixel_mirror_hsi(input_normalize, patch=2):
    ''' 根据patch对，原HSI数据进行padding操作
    为了方便描述，将HSI影像表述为 图像
    将HSI影像中相邻的像素合并作为输入，因此需要对原图像进行padding，进而扩展图像，否则有些像素点可能找不到相邻像素点
    拓展的准则是 对原图像 进行上下左右填充，填充的内容为相邻边界处的像素值
    比如：
        原图像为：[1, 2], 那么向右padding=2时，则产生的镜像为：[1, 2, 2, 2]， 全都用 2 向右填充
    :param input_normalize: 表示经过标准化后的 原图像
    :param patch: 根据 相邻多少个像素 合为一起 来计算 padding多少圈  padding=patch//2
    :return: 返回padding后的镜像input_normalize数据
    '''
    height,width,band = input_normalize.shape
    padding=patch//2
    mirror_input=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_input[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_input[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_input[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_input[i,:,:]=mirror_input[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_input[height+padding+i,:,:]=mirror_input[height+padding-1-i,:,:]

    return mirror_input, padding


def pixel_mirror_hsi_2(input_normalize, pixel_patch):
    l = input_normalize.shape[2]        # 获取通道数
    temp = input_normalize[:, :, 0]
    pad_width = np.floor(pixel_patch / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    mirror_input = np.empty((m2, n2, l), dtype='float32')

    for i in range(l):
        temp = input_normalize[:, :, i]
        pad_width = np.floor(pixel_patch / 2)
        pad_width = np.int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        mirror_input[:, :, i] = temp2

    return mirror_input, pad_width  # 返回padding后的图像，同时返回padding的宽度，用于定位像素新的位置



def gain_pixel_patch(temp_zeros, mirror_input, pos, patch):
    '''
    生成pixel_patch数据
    :param temp_zeros: [patch, patch, band]
    :param mirror_input: padding后的镜像数据
    :param pos: [pos1, pos2] 像素坐标值
    :param patch: 多少个pixel为一组数据
        patch为奇数时, 如patch=5:
            a b c d e   此时 m 为待训练的像素，同时 m 作为新的图像的中心，周围的其他像素为要patch的像素
            f g h i j
            k l m n o
            p q r s t
            u v w x y
            首先patch//2 = 5//2 =2，因 m 周围有2圈数据，中心偏移量为 2
            现在将中心m 偏移到 a的位置，这样便很容易通过两层循环得到所有 patch中像素点的坐标
        patch为偶数时，如patch=4:
            a b c d    此时 f 为待训练的像素，同时 f 作为新的图像的中心，周围的其他像素为要patch的像素
            e f g h
            i j k l
            m n o p
            首先patch//2 = 4//2 =2，但 f 周围有1圈数据，中心偏移量为 2-1=1
            现在将中心m 偏移到 a的位置，这样便很容易通过两层循环得到所有 patch中像素点的坐标
    :return:
    '''
    # ================= 生成pixel_patch数据 ==================
    offset = patch//2 if patch%2!=0 else (patch//2)-1  # patch为奇偶时中心的偏移量不同
    new_center_pos = (pos[0]-offset, pos[1]-offset)     # 设定新的像素中心，原来是以训练像素为中心，只是形式上变了，便于索引，表达的含义其实不变
    for x in range(patch):
        for y in range(patch):
            temp_zeros[x, y, :] = mirror_input[new_center_pos[0]+x, new_center_pos[1]+y, :]
            # temp_zeros[x:x+1, y:y+1, :] = mirror_input[new_center_pos[0]+x:new_center_pos[0]+x+1, new_center_pos[1]+y:new_center_pos[1]+y+1, :]

    return temp_zeros


def gain_pixel_patch_data(index, pos_label, padding, mirror_input, pixel_patch):
    _, _, band = mirror_input.shape
    pos, label = pos_label[index]  # 获取指定索引位置的数据 和 标签
    new_pos = pos + padding     # 获取padding后像素新的坐标位置
    # 1. pixel_patch 获取与中心像素相邻的像素
    mirror_hsi = np.zeros((pixel_patch, pixel_patch, band))
    temp_zeros = mirror_hsi  # 产生占位的数据，方便patch，用0填充
    mirror_hsi = mirror_input[(new_pos[0]-padding):(new_pos[0]+padding), (new_pos[1]-padding):(new_pos[1]+padding), :] if pixel_patch>1 else mirror_input[(new_pos[0]-padding):(new_pos[0]+1), (new_pos[1]-padding):(new_pos[1]+1), :]
    # if pixel_patch % 2 != 0:
    #     mirror_hsi = mirror_input[(new_pos[0]-padding):(new_pos[0]+padding+1), (new_pos[1]-padding):(new_pos[1]+padding+1), :]
    # else:
    #     mirror_hsi = mirror_input[(new_pos[0]-padding):(new_pos[0]+padding), (new_pos[1]-padding):(new_pos[1]+padding), :]

    mirror_hsi = gain_pixel_patch(temp_zeros, mirror_input, new_pos, pixel_patch) if pixel_patch > 1 else mirror_hsi  # 得到与中心像素相邻的像素

    data = torch.from_numpy(mirror_hsi.transpose((2, 0, 1))).type(torch.FloatTensor)  # [c, h, w]

    data_info = {
        'pos': pos,
        'pixel_ori_data': torch.from_numpy(mirror_hsi).type(torch.FloatTensor)
    }

    return data, label, data_info


def gain_band_pixel_patch_data(index, pos_label, padding, mirror_input, band_patch=1, pixel_patch=1):
    '''band_pixel_patch
        pixel_patch 表示中心像素周围的 patch长 和 patch宽 个像素组合在一起, pixel_patch目的是获取与中心像素相邻的像素
        band_patch 处理方式与 gain_band_patch函数一致，对单个像素进行通道上的变换
        比如 pixel_patch=3, band_patch=3, band=200:
            首相找到中心像素周围的8个像素，共9个像素
            然后对这个9个像素 依次使用 gain_band_patch
            最后将这个9个feature_map在通道维上进行拼接
            故最终的形状为: [9*3, 15, 15]
    '''
    _, _, band = mirror_input.shape
    pos, label = pos_label[index]  # 获取指定索引位置的数据 和 标签
    new_pos = pos + padding
    # 1. pixel_patch 获取与中心像素相邻的像素
    mirror_hsi = np.zeros((pixel_patch, pixel_patch, band))
    temp_zeros = mirror_hsi  # 产生占位的数据，方便patch，用0填充
    mirror_hsi = mirror_input[(new_pos[0]-padding):(new_pos[0]+padding), (new_pos[1]-padding):(new_pos[1]+padding), :]
    # =================进行镜像操作==================
    # 只有当patch=1时才需要进行patch操作，否则使用初始数据即可
    mirror_hsi = gain_pixel_patch(temp_zeros, mirror_input, new_pos, pixel_patch) if pixel_patch > 1 else mirror_hsi    # 得到与中心像素相邻的像素 [2, 2, 200]

    # 2. band_patch 将1维通道 展开为 2维特征图
    height_and_width = np.sqrt(band)
    # 只有当无法正好开放时，才需要+1操作
    feature_map_size = int(height_and_width) + 1 if height_and_width > np.floor(height_and_width) else int(np.floor(height_and_width))  # 将一维band映射到二维 feature_map中
    final_mirror_hsi = np.zeros((band_patch*pow(pixel_patch,2), feature_map_size, feature_map_size))   # 存放最终输出的数据，通道 是由 band_patch和pixel_patch共同决定的
    index_i = 0   # 用于记录当前是哪一个 pixel
    index_j = 1   # 用于记录当前是哪一个 pixel
    for x in range(pixel_patch):
        for y in range(pixel_patch):
            mirror_hsi_zeros = np.zeros((1, 1, feature_map_size * feature_map_size))    # 将一维band映射到二维 feature_map中，同时在末尾补0
            temp_zeros = np.zeros((1, 1, feature_map_size * feature_map_size))    # temp_zeros用于patch时暂时存放变形后的mirror_hsi数据
            mirror_hsi_zeros[:, :, :band] = mirror_hsi[x, y, :]    # 找到原数据对应索引位置处的data cube
            temp_mirror_hsi = mirror_hsi_zeros    # temp_mirror_hsi用于记录通道仍为1维状态的数值，方便之后进行patch
            mirror_hsi_zeros = mirror_hsi_zeros.reshape((1, feature_map_size, feature_map_size))
            # ================进行镜像操作==========================
            mirror_hsi_zeros = gain_band_patch(mirror_hsi_zeros, temp_mirror_hsi, temp_zeros, band, feature_map_size, band_patch) if band_patch>1 else mirror_hsi_zeros  # [band_patch, feature_map_size, feature_map_size]
            final_mirror_hsi[index_i*band_patch:index_j*band_patch, :, :] = mirror_hsi_zeros
            index_i += 1  # 索引加一
            index_j += 1  # 索引加一

    data = torch.from_numpy(final_mirror_hsi).type(torch.FloatTensor)  # [c, h, w]

    data_info = {
        'pos': pos,
        'band_ori_data': torch.from_numpy(mirror_input[new_pos[0], new_pos[1], :].reshape(1, 1, -1)).type(torch.FloatTensor),
        'pixel_ori_data': torch.from_numpy(mirror_hsi).type(torch.FloatTensor),
        'feature_map_size': feature_map_size
    }

    return data, label, data_info


# ====================================== 数据处理方法 2 ==========================================
def process_all_data_band_patch(pos_label, input_normalize, patch=1):
    _, _, band = input_normalize.shape
    height_and_width = np.sqrt(band)
    # 只有当无法正好开放时，才需要+1操作
    feature_map_size = int(height_and_width) + 1 if height_and_width > np.floor(height_and_width) else int(np.floor(height_and_width))  # 将一维band映射到二维 feature_map中
    all_data = np.zeros(shape=(len(pos_label), patch, feature_map_size, feature_map_size))
    all_labels = []
    for index in range(len(pos_label)):
        data, label, data_info = gain_band_patch_data(index, pos_label, input_normalize, patch)
        all_data[index, :, :, :] = data
        all_labels.append(label)

    all_data = torch.from_numpy(all_data).type(torch.FloatTensor)
    return all_data, all_labels


def process_all_data_pixel_patch(pos_label, padding, mirror_input, pixel_patch):
    _, _, band = mirror_input.shape
    all_data = np.empty((len(pos_label), band, pixel_patch, pixel_patch))
    all_labels = []

    for idx, (pos, label) in enumerate(pos_label):
        new_pos = pos + padding
        if pixel_patch % 2 != 0:
            mirror_hsi = mirror_input[(new_pos[0]-padding):(new_pos[0]+padding+1), (new_pos[1]-padding):(new_pos[1]+padding+1), :]
        else:
            mirror_hsi = mirror_input[(new_pos[0]-padding):(new_pos[0]+padding), (new_pos[1]-padding):(new_pos[1]+padding), :]
        mirror_hsi = mirror_hsi.transpose((2, 0, 1))
        all_data[idx, :, :, :] = mirror_hsi
        all_labels.append(label)

    all_data = torch.from_numpy(all_data).type(torch.FloatTensor)
    return all_data, all_labels


def process_all_data_band_pixel_patch(pos_label, input_normalize, mirror_input, band_patch=1, pixel_patch=1):
    _, _, band = input_normalize.shape
    height_and_width = np.sqrt(band)
    # 只有当无法正好开放时，才需要+1操作
    feature_map_size = int(height_and_width) + 1 if height_and_width > np.floor(height_and_width) else int(np.floor(height_and_width))  # 将一维band映射到二维 feature_map中
    all_data = np.zeros(shape=(len(pos_label), band*pow(pixel_patch, 2), feature_map_size, feature_map_size))
    all_labels = []
    for index in range(len(pos_label)):
        data, label, data_info = gain_band_pixel_patch_data(index, pos_label, input_normalize, mirror_input, band_patch, pixel_patch)
        all_data[index, :, :, :] = data
        all_labels.append(label)

    all_data = torch.from_numpy(all_data).type(torch.FloatTensor)
    return all_data, all_labels



def process_all_data_pixel_patch_2(input_normalize, pixel_patch, label):
    [m, n, l] = np.shape(input_normalize)
    temp = input_normalize[:, :, 0]
    pad_width = np.floor(pixel_patch / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x2 = np.empty((m2, n2, l), dtype='float32')
    for i in range(l):
        temp = input_normalize[:, :, i]
        pad_width = np.floor(pixel_patch / 2)
        pad_width = np.int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2[:, :, i] = temp2

    [ind1, ind2] = np.where(label != 0)
    Num = len(ind1)  # 695
    Patch = np.empty((Num, l, pixel_patch, pixel_patch), dtype='float32')  # [695, 200, 16, 16]
    Label = np.empty(Num)  # 695
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        if pixel_patch % 2 != 0:    # pixel_patch 为奇数
            patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        else:  # pixel_patch 为偶数
            patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]  # [16, 16, 200]
        patch = np.reshape(patch, (pixel_patch * pixel_patch, l))  # [256, 200]
        patch = np.transpose(patch)  # [200, 256]
        patch = np.reshape(patch, (l, pixel_patch, pixel_patch))  # [200, 16, 16]
        Patch[i, :, :, :] = patch
        patchlabel = label[ind1[i], ind2[i]]
        Label[i] = patchlabel - 1

    return torch.from_numpy(Patch), torch.from_numpy(Label).long()