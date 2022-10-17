# -*- coding: utf-8 -*-
# date: 2022/3/17 22:45
# Project: 毕业设计
# File Name: train_options.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import os
import yaml
import argparse


def train_options(settings):
    parser = argparse.ArgumentParser('HSI')
    # 训练前模型配置
    parser.add_argument('--data_name', type=str, default=settings['data_name'], choices=['Indian', 'Pavia', 'Houston'], help='dataset to use')
    parser.add_argument('--train', type=bool, default=settings['train'], help='True means Train, False means Test')
    parser.add_argument('--seed', type=int, default=settings['seed'], help='number of seed')
    parser.add_argument('--device', type=str, default=settings['device'], choices=['cuda:0', 'cpu'], help='train device')
    parser.add_argument('--epochs', type=int, default=settings['epochs'], help='training epochs')
    parser.add_argument('--batch_size', type=int, default=settings['batch_size'], help='dataloader batch_size')
    parser.add_argument('--swin_type', type=str, default=settings['swin_type'], choices=['tiny', 'small', 'base'], help='选择加载swin-transformer模型的类型')
    parser.add_argument('--process_data', type=str, default=settings['process_data'], choices=['all', 'batch'], help='all 训练前一次性处理所有数据, batch 每一个batch处理数据一次')
    # 学习率调整
    parser.add_argument('--learn_rate', type=float, default=settings['learn_rate'], help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=settings['weight_decay'], help='weight_decay')
    parser.add_argument('--scheduler_name', type=str, default=settings['scheduler_name'], choices=['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'CosineAnnealingLR'], help='学习率衰减方式')
    parser.add_argument('--scheduler_gamma', type=float, default=settings['scheduler_gamma'], help='StepLR参数 scheduler_gamma')  # StepLR参数
    parser.add_argument('--scheduler_step_size', type=int, default=settings['scheduler_step_size'], help='StepLR参数 scheduler_step_size')
    parser.add_argument('--cawr_t_0', type=int, default=settings['cawr_t_0'], help='CosineAnnealingWarmRestart参数, T_0控制余弦多少epoch一次震荡')
    parser.add_argument('--cawr_t_mult', type=int, default=settings['cawr_t_mult'], help='CosineAnnealingWarmRestart参数, T_mult就是重启之后因子 T_0=T_0*T_mult')
    # 模型保存
    parser.add_argument('--save_log', type=bool, default=settings['save_log'], help='save the train, valid or test log')
    parser.add_argument('--log_dir', type=str, default=settings['log_dir'], help='训练过程中模型保存的文件夹')
    parser.add_argument('--exp_name_dir', type=str, default=settings['exp_name_dir'], help='该次实验及模型保存的文件夹,path=log_dir/exp_name_dir')
    parser.add_argument('--result_dir', type=str, default=settings['result_dir'], help='best model在测试集上的结果,path=log_dir/exp_name_dir/result_dir')
    parser.add_argument('--use_tensorboard', type=bool, default=settings['use_tensorboard'], help='是否启用tensorboard')
    parser.add_argument('--tensorboard_dir', type=str, default=settings['tensorboard_dir'], help='tensorboard结果保存的目录,path=log_dir/exp_name_dir/tensorboard_dir')
    # 是否启用训练时进行验证
    parser.add_argument('--use_val_step', type=bool, default=settings['use_val_step'], help='是否在训练的时候进行验证集上的检验')
    parser.add_argument('--val_step', type=int, default=settings['val_step'], help='每隔多少epoch进行一次验证')
    # 是否启用early stopping
    parser.add_argument('--use_early_stopping', type=bool, default=settings['use_early_stopping'], help='是否使用早停')
    parser.add_argument('--early_stopping', type=int, default=settings['early_stopping'], help='连续多少epoch acc不发生相差小于0.5%就停止训练')
    # 数据加载方式
    parser.add_argument('--mode', type=str, default=settings['mode'], choices=['band_patch', 'pixel_patch'], help='how the dataset is processed')
    parser.add_argument('--band_patch', type=int, default=settings['band_patch'], help='band patch size,当mode为含有band时生效')
    parser.add_argument('--pixel_patch', type=int, default=settings['pixel_patch'], help='the pixel patch size,当mode为含有pixel时生效')
    # 在band维进行数据降维
    parser.add_argument('--use_dimension_reduction', type=bool, default=settings['use_dimension_reduction'], help='是否使用数据降维')
    parser.add_argument('--dimension_mode', type=str, default=settings['dimension_mode'], choices=['FA', 'PCA'], help='因子分析和主成分分析')
    parser.add_argument('--numcomponents', type=int, default=settings['numcomponents'], help='降维后的特征数')
    # 使用 特征金字塔 和 残差结构
    parser.add_argument('--use_fp', type=bool, default=settings['use_fp'], help='是否使用特征金字塔结构')
    parser.add_argument('--fp_name', type=str, default=settings['fp_name'], choices=['SPP', 'ASPP', 'FPN', 'PANet'], help='可选的FP结构')
    parser.add_argument('--use_resblock', type=bool, default=settings['use_resblock'], help='是否使用残差结构')
    # 选择 损失函数
    parser.add_argument('--loss_function', type=str, default=settings['loss_function'], choices=['CE_loss', 'focal_loss'], help='交叉熵损失 和 focal损失')
    parser.add_argument('--alpha_mode', type=str, default=settings['alpha_mode'], choices=['auto', 'handle', None], help='focal loss中alpha的获取方式,auto为自动根据训练数据类别比例获取,handle为人为根据数据类别比例手动划分,None为不划分alpha默认均为1')
    # 使用通道注意力机制
    parser.add_argument('--use_cam', type=bool, default=settings['use_cam'], help='通道注意力机制')
    # 使用 分支 空间 通道注意力机制
    parser.add_argument('--use_attention', type=bool, default=settings['use_attention'], help='是否在模型开头使用空间或通道注意力机制')
    parser.add_argument('--change_channel', type=bool, default=settings['change_channel'], help='是否在attention后改变通道数，在不使用fusion时使用')
    parser.add_argument('--use_fusion', type=bool, default=settings['use_fusion'], help='是否在使用attention后进行特征融合')


    opt = parser.parse_args()

    return opt


def load_settings(file):
    settings_dir = os.path.dirname(os.path.realpath(__file__))
    settings_path = os.path.join(settings_dir, 'settings', file)

    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = yaml.load(f.read(), Loader=yaml.FullLoader)

    # print(settings)
    return settings


def show_options(opt):
    for k, v in vars(opt).items():
        print(k.rjust(25)+": "+str(v))


if __name__ == '__main__':
    settings = load_settings('band-patch-1.yaml')
    opt = train_options(settings)
    show_options(opt)