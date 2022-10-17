# -*- coding: utf-8 -*-
# date: 2022/3/17
# Project: 毕业设计
# File Name: base_options.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import argparse


def base_options():
    parser = argparse.ArgumentParser('HSI')
    # 训练前模型配置
    parser.add_argument('--data_name', type=str, default='Indian', choices=['Indian', 'Pavia', 'Houston'], help='dataset to use')
    parser.add_argument('--train', type=bool, default=True, help='True means Train, False means Test')
    parser.add_argument('--seed', type=int, default=2, help='number of seed')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'], help='train device')
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='dataloader batch_size')
    parser.add_argument('--swin_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='选择加载swin-transformer模型的类型')
    parser.add_argument('--process_data', type=str, default='all', choices=['all', 'batch'], help='all 训练前一次性处理所有数据, batch 每一个batch处理数据一次')
    # 学习率调整
    parser.add_argument('--learn_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--scheduler_name', type=str, default='StepLR', choices=['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'CosineAnnealingLR'], help='学习率衰减方式')
    parser.add_argument('--scheduler_gamma', type=float, default=0.9, help='StepLR参数 scheduler_gamma')  # StepLR参数
    parser.add_argument('--scheduler_step_size', type=int, default=5, help='StepLR参数 scheduler_step_size')
    parser.add_argument('--cawr_t_0', type=int, default=10, help='CosineAnnealingWarmRestart参数, T_0控制余弦多少epoch一次震荡')
    parser.add_argument('--cawr_t_mult', type=int, default=2, help='CosineAnnealingWarmRestart参数, T_mult就是重启之后因子 T_0=T_0*T_mult')
    # 模型保存
    parser.add_argument('--save_log', type=bool, default=False, help='save the train, valid or test log')
    parser.add_argument('--log_dir', type=str, default='log', help='训练过程中模型保存的文件夹')
    parser.add_argument('--exp_name_dir', type=str, default='train', help='该次实验及模型保存的文件夹,path=log_dir/exp_name_dir')
    parser.add_argument('--result_dir', type=str, default='result', help='best model在测试集上的结果,path=log_dir/exp_name_dir/result_dir')
    parser.add_argument('--use_tensorboard', type=bool, default=True, help='是否启用tensorboard')
    parser.add_argument('--tensorboard_dir', type=str, default='train_log', help='tensorboard结果保存的目录,path=log_dir/exp_name_dir/tensorboard_dir')
    # 是否启用训练时进行验证
    parser.add_argument('--use_val_step', type=bool, default=True, help='是否在训练的时候进行验证集上的检验')
    parser.add_argument('--val_step', type=int, default=1, help='每隔多少epoch进行一次验证')
    # 是否启用early stopping
    parser.add_argument('--use_early_stopping', type=bool, default=True, help='是否使用早停')
    parser.add_argument('--early_stopping', type=int, default=3, help='连续多少epoch acc不发生相差小于0.5%就停止训练')
    # 数据加载方式
    parser.add_argument('--mode', type=str, default='band_patch', choices=['band_patch', 'pixel_patch'], help='how the dataset is processed')
    parser.add_argument('--band_patch', type=int, default=3, help='band patch size,当mode为含有band时生效')
    parser.add_argument('--pixel_patch', type=int, default=3, help='the pixel patch size,当mode为含有pixel时生效')
    # 在band维进行数据降维
    parser.add_argument('--use_dimension_reduction', type=bool, default=True, help='是否使用数据降维')
    parser.add_argument('--dimension_mode', type=str, default='FA', choices=['FA', 'PCA'], help='因子分析和主成分分析')
    parser.add_argument('--numcomponents', type=int, default=10, help='降维后的特征数')
    # 使用 特征金字塔 和 残差结构
    parser.add_argument('--use_fp', type=bool, default=False, help='是否使用特征金字塔结构')
    parser.add_argument('--fp_name', type=str, default='SPP', choices=['SPP', 'ASPP', 'FPN', 'PANet'], help='可选的FP结构')
    parser.add_argument('--use_resblock', type=bool, default=True, help='是否使用残差结构')
    # 选择 损失函数
    parser.add_argument('--loss_function', type=str, default='CE_loss', choices=['CE_loss', 'focal_loss'], help='交叉熵损失 和 focal损失')
    parser.add_argument('--alpha_mode', type=str, default=None, choices=['auto', 'handle', None], help='focal loss中alpha的获取方式,auto为自动根据训练数据类别比例获取,handle为人为根据数据类别比例手动划分,None为不划分alpha默认均为1')
    # 使用通道注意力机制
    parser.add_argument('--use_cam', type=bool, default=True, help='通道注意力机制')
    # 使用 分支 空间 通道注意力机制
    parser.add_argument('--use_attention', type=bool, default=False, help='是否在模型开头使用空间或通道注意力机制')
    parser.add_argument('--change_channel', type=bool, default=False, help='是否在attention后改变通道数，在不使用fusion时使用')
    parser.add_argument('--use_fusion', type=bool, default=True, help='是否在使用attention后进行特征融合')


    opt = parser.parse_args()