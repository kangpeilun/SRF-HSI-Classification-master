# -*- coding: utf-8 -*-
# date: 2022/2/13 15:20
# Project: 毕业设计
# File Name: utils.py
# Description: 工具模块
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import argparse
import sys
import os
import random
import numpy as np
import torch


def same_seeds(seed):
    '''
    将随机种子设置为某个值 以实现可重复性
    :param seed:
    :return:
    '''
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子，以使得结果是确定的
        torch.cuda.manual_seed_all(seed)  # 为所有的 GPU 设置种子用于生成随机数，以使得结果是确定的

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_dir(dir_path):
    '''检测路径是否存在，并创建对应文件夹'''
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def save_log(save_dir_path):
    '''用于记录训练过程中print的输出，将log保存到对应save_dir_path文件夹中'''
    filename = 'train_log.txt'
    file_path = os.path.join('.\\log',save_dir_path,filename)
    output_file = open(file_path, 'w', encoding='utf-8')
    sys.stdout = output_file

    return output_file


def make_all_dirs(opt):
    '''创建必要的文件夹
    log_dir--|
        exp_name_dir--|
                best_model.pth
                last_model.pth
                result_dir--|
                        xxx.png
                        xxx.png
          tensorboard_dir--|
                        xxxxxx
    '''
    log_dir = os.path.join(opt.log_dir)
    exp_name_dir = os.path.join(opt.log_dir, opt.exp_name_dir)
    result_dir = os.path.join(opt.log_dir, opt.exp_name_dir, opt.result_dir)
    tensorboard_dir = os.path.join(opt.log_dir, opt.exp_name_dir, opt.tensorboard_dir) if opt.use_tensorboard else None

    for dir in [log_dir, exp_name_dir, result_dir, tensorboard_dir]:
        if dir is not None:
            check_dir(dir)

    return log_dir, exp_name_dir, result_dir, tensorboard_dir