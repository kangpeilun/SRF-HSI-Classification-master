# -*- coding: utf-8 -*-
# date: 2022/2/13 10:27
# Project: 毕业设计
# File Name: dataset.py
# Description: 获取模型训练所需的数据集
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import torch
from torch.utils.data import DataLoader, Dataset

import argparse
import numpy as np

from utils.dataset import chooose_train_and_test_point, get_data, normalize, pixel_mirror_hsi, pixel_mirror_hsi_2, \
                    gain_band_patch_data, gain_pixel_patch_data, gain_band_pixel_patch_data, choose_alpha, \
                    process_all_data_band_patch, process_all_data_pixel_patch, process_all_data_band_pixel_patch, process_all_data_pixel_patch_2
from utils.dimension_reduction import applyFA, applyPCA, analysis_of_numcomponents, draw_fa_map


class HSIDataset(Dataset):
    def __init__(self, opt, data_name, mode='band_patch', band_patch=1, pixel_patch=1, train='train'):
        '''
        :param data_name: 数据名 ['Indian', 'Pavia', 'Houston']
        :param mode: 选择数据处理的方式 ['band_patch', 'pixel_patch']
                band_patch: 将训练像素的通道展平为一个特征图，(假设有200个通道)
                当patch=1时 数据shape为(15, 15, 1)，patch=2时 shape为(15, 15, 2)，以此类推
                假设特征图大小为2x2，patch=2，下面时两张堆叠的特征图：
                    1 2    —>   2 3     shape: [2, 2, 2]
                    3 4         4 1
                pixel_patch: 将训练像素周围像素进行打组合为一组数据输入到网络进行训练，
                patch=1时 数据的shape为(1, 1, 200)，patch=2时 shape为(2, 2, 200)，以为类推
                假设1号点为要训练的像素，patch=2：
                    1 2     此时 1，2，3，4号像素点组成一组训练数据 shape:[2, 2, 200]
                    3 4
                假设1号点为要训练的像素，patch=3时(此时，需要padding一圈数据)：
                    2 3 4   此时 1，2，3，4，5，6，7，8，9号像素点组成一组训练数据 shape:[3, 3, 200]
                    5 1 6
                    7 8 9
        :param patch: 多少个band或pixel为一组
        :param train: 默认为True，生成用于训练的数据
        '''
        super(HSIDataset, self).__init__()
        self.opt = opt      # 全局配置信息
        self.data_name = data_name
        self.mode = mode
        self.pixel_patch = pixel_patch
        self.band_patch = band_patch
        self.train = train
        # ====================准备数据集======================
        self.input, self.TR, self.TE, self.label, self.num_classes, self.color_matrix = get_data(data_name)
        self.input_normalize = normalize(self.input)  # 将 input数据 在 光谱维标准化, 真正用于训练的数据
        # ===================使用因子分析=====================
        if opt.use_dimension_reduction:
            if opt.dimension_mode == 'FA':
                # print('Using FA.')
                self.input_normalize = applyFA(self.input_normalize, opt.numcomponents)   # [145, 145, 200] -> [145, 145, numComponents]
            elif opt.dimension_mode == 'PCA':
                # print('Using PCA')
                self.input_normalize = applyPCA(self.input_normalize, opt.numcomponents)

        # 使用因子分析后，数据的通道数会发生改变，故将数据信息放在FA和PCA之后获取
        # self.mirror_input, self.padding = pixel_mirror_hsi(self.input_normalize, self.pixel_patch)    # 获取padding后的数据, 针对pixel-patch 和 band-pixel-patch
        self.mirror_input, self.padding = pixel_mirror_hsi_2(self.input_normalize, self.pixel_patch)    # 获取padding后的数据, 针对pixel-patch 和 band-pixel-patch
        self.data_shape = self.input_normalize.shape

        # 获取训练集 测试集 标签对应的索引，获取数据基本信息
        self.train_pos_label, self.test_pos_label, self.true_pos_label, \
        self.total_pos_train, self.total_pos_test, self.total_pos_true, \
        self.number_train, self.number_test, self.number_true = chooose_train_and_test_point(
            self.num_classes, self.TR, self.TE, self.label
        )  # 选择训练数据 和 测试数据

        # 使用focal loss时根据 训练集 不同类别的比例设置 alpha
        self.alpha = choose_alpha(self.number_train, opt.alpha_mode, data_name)

        if opt.process_data == 'all':
            self.all_data, self.all_labels = self.process_all_data_before_train_test()        # 数据处理方式, 在训练前一次性将所有数据都处理好


    def process_all_data_before_train_test(self):
        '''在训练和测试前统一将所有的数据处理好'''
        if self.mode == 'band_patch':
            if self.train:
                all_data, all_labels = process_all_data_band_patch(self.train_pos_label, self.input_normalize, self.band_patch)
                return all_data, all_labels
            else:
                all_data, all_labels = process_all_data_band_patch(self.test_pos_label, self.input_normalize, self.band_patch)
                return all_data, all_labels
        elif self.mode == 'pixel_patch':
            if self.train:
                all_data, all_labels = process_all_data_pixel_patch(self.train_pos_label, self.padding, self.mirror_input, self.pixel_patch)
                # all_data, all_labels = process_all_data_pixel_patch_2(self.input_normalize, self.pixel_patch, self.TR)
                return all_data, all_labels
            else:
                all_data, all_labels = process_all_data_pixel_patch(self.test_pos_label, self.padding, self.mirror_input, self.pixel_patch)
                # all_data, all_labels = process_all_data_pixel_patch_2(self.input_normalize, self.pixel_patch, self.TE)
                return all_data, all_labels
        elif self.mode == 'band_pixel_patch':
            if self.train:
                all_data, all_labels = process_all_data_band_pixel_patch(self.train_pos_label, self.input_normalize, self.mirror_input, self.band_patch, self.pixel_patch)
                return all_data, all_labels
            else:
                all_data, all_labels = process_all_data_band_pixel_patch(self.test_pos_label, self.input_normalize, self.mirror_input, self.band_patch, self.pixel_patch)
                return all_data, all_labels

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        if self.opt.process_data == 'all':
            x_train, label = self.all_data[index], self.all_labels[index]
            return x_train, label

        elif self.opt.process_data == 'batch':
            if self.mode == 'band_patch':
                # 产生训练数据
                if self.train == 'train':
                    x_train, label, data_info = gain_band_patch_data(index, self.train_pos_label, self.input_normalize, self.band_patch)
                    return x_train, label, data_info
                # 产生测试数据
                elif self.train == 'test':
                    x_test, label, data_info = gain_band_patch_data(index, self.test_pos_label, self.input_normalize, self.band_patch)
                    return x_test, label, data_info
                elif self.train == 'predict':
                    x_test, label, data_info = gain_band_patch_data(index, self.true_pos_label, self.input_normalize, self.band_patch)
                    return x_test, label, data_info

            elif self.mode == 'pixel_patch':
                if self.train == 'train':
                    x_train, label, data_info = gain_pixel_patch_data(index, self.train_pos_label, self.padding, self.mirror_input, self.pixel_patch)
                    return x_train, label, data_info
                elif self.train == 'test':
                    x_test, label, data_info = gain_pixel_patch_data(index, self.test_pos_label, self.padding, self.mirror_input, self.pixel_patch)
                    return x_test, label, data_info
                elif self.train == 'predict':
                    x_test, label, data_info = gain_pixel_patch_data(index, self.true_pos_label, self.padding, self.mirror_input, self.pixel_patch)
                    return x_test, label, data_info

            elif self.mode == 'band_pixel_patch':
                if self.train == 'train':
                    x_train, label, data_info = gain_band_pixel_patch_data(index, self.train_pos_label, self.padding, self.mirror_input, self.band_patch, self.pixel_patch)
                    return x_train, label, data_info
                elif self.train == 'test':
                    x_test, label, data_info = gain_band_pixel_patch_data(index, self.test_pos_label, self.padding, self.mirror_input, self.band_patch, self.pixel_patch)
                    return x_test, label, data_info
                elif self.train == 'predict':
                    x_test, label, data_info = gain_band_pixel_patch_data(index, self.true_pos_label, self.padding, self.mirror_input, self.band_patch, self.pixel_patch)
                    return x_test, label, data_info

    def __len__(self):
        if self.train == 'train':
            return sum(self.number_train)
        elif self.train == 'test':
            return sum(self.number_test)
        elif self.train == 'predict':
            return sum(self.number_true)


    def collate_fn(self, batch):
        '''
        因band_pixel_patch的特殊性,需单独进行处理
        :param batch: [(x1, label1), (x2, label2), ...]
        :return:
        '''
        if self.mode == 'band_pixel_patch':
            new_batch = []
            for data, label in batch:
                # print(data.shape, label)
                data_list = data.chunk(self.band_patch, dim=0)  # 在第0个维度上将data拆分
                for new_data in data_list:
                    new_batch.append((new_data, label))
            batch = new_batch

        # 将 [tensor(), tensor(), ...] 变为 tensor([], [], []) 的形式, 否则无法使用gpu进行加速
        data_1 = batch[0][0]
        label_1 = batch[0][1]
        new_data = np.array(data_1.unsqueeze(0).numpy())
        new_label = np.array(label_1)
        for data, label in batch[1:]:
            data = data.unsqueeze(0)
            new_data = np.vstack((new_data, data.numpy()))
            new_label = np.append(new_label, label)

        return torch.tensor(new_data).type(torch.FloatTensor), torch.tensor(new_label).type(torch.long)


    def analysis_fa(self):
        '''查看因子分析结果'''
        analysis_of_numcomponents(self.input_normalize)   # 返回因子分析 折线图，便于观察与那些因子有关
        draw_fa_map(self.input_normalize)                 #

    def data_info(self):
        '''输出数据基本信息'''
        print('-' * 20, 'Data Info', '-' * 20)
        print(f'input size of {self.data_name} data: ', self.input.shape)
        print('train data size(TR): ', self.TR.shape)
        print('test data size(TE): ', self.TE.shape)
        print(f'the num_classes of {self.data_name} data: ', self.num_classes)
        print('the color_matrix size: ', self.color_matrix.shape)

        print('num of train: ', self.number_train, sum(self.number_train))
        print('num of test: ', self.number_test, sum(self.number_test))
        print('num of true: ', self.number_true, sum(self.number_true))






def get_dataloader(opt, data_name, mode='band_patch', band_patch=1, pixel_patch=1, train='train', batch_size=8, shuffle=True):
    '''
    生成dataloader，用数据加载
    :param data_name: 数据名 ['Indian', 'Pavia', 'Houston']
    :param mode: 选择数据处理的方式 ['band_patch', 'pixel_patch', band_pixel_patch]
    :param band_patch: 多少个band为一组
    :param pixel_patch: 多少个pixel为一组
    :param train: 默认为True，生成用于训练的数据
    :param batch_size:
    :param shuffle: 是否打乱数据，默认为True训练集需要打乱，测试集为False
    :return:
    '''
    dataset = HSIDataset(opt, data_name, mode, band_patch, pixel_patch, train)
    # 展示数据基本信息
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    if train == 'train':
        dataset.data_info()
        print('-' * 20, 'Start Training', '-' * 20)
        # print(f'gain the dataloader for Swin-Transformer with the setting -> mode={mode}, train={train}')

    return dataloader


if __name__ == '__main__':
    # dataset = HSIDataset(data_name='Indian', mode='band_patch', patch=64, train=True)
    # dataset = HSIDataset(data_name='Indian', mode='pixel_patch', patch=3, train=True)
    # print(dataset[0])
    # dataloader = get_dataloader('Indian', 'band_patch', False, batch_size=64)
    # dataloader = get_dataloader('Indian', 'pixel_patch', True, batch_size=64)
    parser = argparse.ArgumentParser('HSI')
    parser.add_argument('--seed', type=int, default=2, help='number of seed')
    parser.add_argument('--use_dimension_reduction', type=bool, default=False, help='是否使用数据降维')
    parser.add_argument('--dimension_mode', type=str, default='FA', choices=['FA', 'PCA'], help='因子分析和主成分分析')
    parser.add_argument('--numcomponents', type=int, default=10, help='降维后的特征数')
    parser.add_argument('--alpha_mode', type=str, default='handle', choices=['auto', 'handle', None], help='focal loss中alpha的获取方式,auto为自动根据训练数据类别比例获取,handle为人为根据数据类别比例手动划分,None为不划分alpha默认均为1')  # 当alpha=None时 focal loss效果和CE_loss一样
    # 数据处理方式
    parser.add_argument('--mode', type=str, default='pixel_patch',
                        choices=['band_patch', 'pixel_patch', 'band_pixel_patch'],
                        help='how the dataset is processed')  # 切记 如果为band_pixel_patch, 则参与运算的数据的batch_size会变为batch_size*band_patch
    parser.add_argument('--band_patch', type=int, default=3, help='band patch size,当mode为含有band时生效')
    parser.add_argument('--pixel_patch', type=int, default=2, help='the pixel patch size,当mode为含有pixel时生效')
    parser.add_argument('--process_data', type=str, default='batch', choices=['all', 'batch'], help='all 训练前一次性处理所有数据, batch 每一个batch处理数据一次')
    opt = parser.parse_args()

    dataloader = get_dataloader(opt, data_name='Indian', mode='band_pixel_patch', band_patch=2, pixel_patch=2, train='predict', batch_size=60)
    datal = iter(dataloader).next()
    for data, label, data_info in dataloader:
        # x = data[0]
        # y = data[1]
        print(data, label)
        break