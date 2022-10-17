# -*- coding: utf-8 -*-
# date: 2022/3/17
# Project: 毕业设计
# File Name: train_cycle.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from main import Train_and_Test_Swin_model
from options.train_options import train_options, load_settings, show_options
from utils.utils import save_log

from indian_config import indian_config_file
from houston_config import houston_config_file
from pavia_config import pavia_config_file


def load_new_opt(opt, file):
    # ============== new settings ===============
    # 对部分设置进行全局修改
    file_dir = file.rsplit('/', 2)[-2]
    if 'indian' in file:
        # opt.seed = 0
        opt.batch_size = 64
        opt.epochs = 300
        opt.val_step = 1
    elif 'pavia' in file:
        # opt.seed = 0
        opt.batch_size = 64
        opt.epochs = 300
        opt.val_step = 1
    elif 'houston' in file:
        # opt.seed = 0
        opt.batch_size = 64
        opt.epochs = 300
        opt.val_step = 1

    # opt.exp_name_dir = file.rsplit('/', 1)[-1].replace('.yaml', '')  # 确保待保存文件夹不会出错

    # val while train
    opt.use_val_step = True
    # opt.val_step = 2
    # early stopping
    opt.use_early_stopping = False
    opt.early_stopping = 4
    # scheduler
    opt.scheduler_name = 'CosineAnnealingWarmRestarts'  # ['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'CosineAnnealingLR']  # CosineAnnealingWarmRestarts效果最好
    opt.scheduler_gamma = 0.2  # StepLR时使用
    # 数据加载方式
    # opt.mode = 'band_pixel_patch'
    # opt.band_patch = 14
    # opt.pixel_patch = 7
    # # dimension_reduction
    opt.use_dimension_reduction = False
    # opt.dimension_mode = 'FA'  # ['FA', 'PCA']
    # opt.numcomponents = 2
    # ============== new settings ===============

    return opt


def train_cycle():
    config_file_list = indian_config_file + pavia_config_file + houston_config_file   # 将多个配置文件数组合并
    for file in indian_config_file:
    # for file in pavia_config_file:
    # for file in houston_config_file:
    # for file in config_file_list:
        settings = load_settings(file)
        opt = train_options(settings)
        # ============== new settings ===============
        opt = load_new_opt(opt, file)
        # ============== new settings ===============
        train_and_test_model = Train_and_Test_Swin_model(opt)

        print('\n'+'-'*20, file, '-'*20)
        show_options(opt)
        if opt.save_log:
            # 将训练过程中的print输出保存到result_dir文件夹中对应的txt文件中
            # 此时不会在 terminal中显示print输出
            output_file = save_log(train_and_test_model.result_dir)
            train_and_test_model.train_model()

            output_file.close()
        else:
            train_and_test_model.train_model()


if __name__ == '__main__':
    train_cycle()