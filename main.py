# -*- coding: utf-8 -*-
# date: 2022/2/13 20:04
# Project: 毕业设计
# File Name: main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

import os
import numpy as np
import argparse

from model.SwinTransformer import swin_tiny_patch4_window7_224
from model.HSISwinNet import MyNet
from dataset import get_dataloader
from utils.utils import save_log, same_seeds, make_all_dirs
from utils.main import train_one_epoch, valid_one_epoch, predict_one_epoch, \
                        output_metric, show_result, save_one_model, write_txt, \
                        get_scheduler, get_loss
from utils.draw_predict_data import draw_data


class Train_and_Test_Swin_model():
    def __init__(self, opt):
        self.opt = opt  # 全局配置信息
        # -------------------创建必要的文件夹---------------
        self.log_dir, self.exp_name_dir, self.result_dir, self.tensorboard_dir = make_all_dirs(opt)
        # ------------ -设置随机种子，使结果可以复现----------
        same_seeds(opt.seed)
        # -----------------启用tensorboard----------------
        if opt.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        # -------------------实例化部分--------------------
        # 加载dataloader
        self.train_dataloader = get_dataloader(opt, opt.data_name, opt.mode, opt.band_patch, opt.pixel_patch, 'train', opt.batch_size)
        self.valid_dataloader = get_dataloader(opt, opt.data_name, opt.mode, opt.band_patch, opt.pixel_patch, 'test', opt.batch_size, False)  # 验证集和测试集是一样的
        self.predict_dataloader = get_dataloader(opt, opt.data_name, opt.mode, opt.band_patch, opt.pixel_patch, 'predict', opt.batch_size, False)

        self.num_calsses = self.train_dataloader.dataset.num_classes
        self.alpha = self.train_dataloader.dataset.alpha
        self.data_shape = self.train_dataloader.dataset.data_shape
        self.positions_train = self.train_dataloader.dataset.total_pos_train
        self.positions_test = self.train_dataloader.dataset.total_pos_test

        # 根据数据加载方式的不同，加载对应的Swin模型
        if opt.mode == 'band_patch':
            # band_patch 数据处理方式，模型的通道数和patch一致
            # self.swin_model = swin_tiny_patch4_window7_224(in_chans=opt.band_patch, num_classes=self.num_calsses).to(opt.device)  # in_chans: band_patch
            self.swin_model = MyNet(opt=opt, band=self.data_shape[2], in_chans=opt.band_patch, num_classes=self.num_calsses).to(opt.device)  # in_chans: band_patch
        elif opt.mode == 'band_pixel_patch':
            # pixel_patch 和 band_pixel_patch 数据处理方式，模型的通道数和数据的通道数一致
            # self.swin_model = swin_tiny_patch4_window7_224(in_chans=self.data_shape[2], num_classes=self.num_calsses).to(opt.device)  # in_chans: band
            self.swin_model = MyNet(opt=opt, band=self.data_shape[2], in_chans=opt.band_patch*pow(opt.pixel_patch,2), num_classes=self.num_calsses).to(opt.device)  # in_chans: band
        elif opt.mode == 'pixel_patch':
            self.swin_model = MyNet(opt=opt, band=self.data_shape[2], in_chans=self.data_shape[2], num_classes=self.num_calsses).to(opt.device)  # in_chans: band

        # 加载损失函数
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = get_loss(opt, self.num_calsses, self.alpha)
        # Adam优化器
        self.optimizer = optim.Adam(self.swin_model.parameters(), lr=opt.learn_rate, weight_decay=opt.weight_decay)
        # 学习率调整
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.scheduler_step_size, gamma=opt.scheduler_gamma)
        self.scheduler = get_scheduler(opt, self.optimizer)

    def train_model(self):
        best_loss = np.inf
        last_acc = 0  # 用于记录上一个epoch的acc
        best_acc = 0.
        best_epoch = 0
        best_OA = 0.
        best_AA_mean = 0.
        best_Kappa = 0.
        best_AA = 0.

        count_stop_epoch = 0    # stop计数，设置连续多少epoch之后,某个指标不发生变化 就停止训练
        self.swin_model.train()
        for epoch in range(1, self.opt.epochs+1):
            # ===============train model=================
            train_loss, train_acc, tar, pre = train_one_epoch(self.swin_model, self.train_dataloader, self.opt.device, self.criterion, self.optimizer)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.scheduler.step()  # 学习率更新

            if self.opt.use_early_stopping:   # 使用早停
                if train_acc - last_acc > 0.5:
                    count_stop_epoch = epoch
                else:
                    if (epoch - count_stop_epoch) % self.opt.early_stopping == 0:
                        print('Epoch:{:03d}\t train_loss:{:.4f}\t train_acc:{:.4f}\t lr:{:.5f}'.format(epoch, train_loss, train_acc, lr))
                        OA, AA_mean, Kappa, AA = self.valid_model()    # 满足early stop条件，则对模型进行测试
                        write_txt(epoch, OA, AA_mean, Kappa, AA, self.opt, self.result_dir)
                        show_result(epoch, OA, AA_mean, Kappa, AA, self.opt)
                        save_one_model(self.swin_model, self.optimizer, self.exp_name_dir, 'early_model')
                        print(f'Early Stopping, Because acc hast update for {self.opt.early_stopping} epochs!')
                        break
                last_acc = train_acc  # 用于记录上一个epoch的acc

            if self.opt.use_val_step and epoch % self.opt.val_step == 0:  # 使用验证集
                OA, AA_mean, Kappa, AA = self.valid_model()
                if OA > best_OA:
                    best_loss = train_loss
                    best_acc = train_acc
                    best_epoch = epoch
                    best_OA = OA
                    best_AA_mean = AA_mean
                    best_Kappa = Kappa
                    best_AA = AA
                    write_txt(best_epoch, OA, AA_mean, Kappa, AA, self.opt, self.result_dir)  # 因为验证机和测试集 是一样的，故每次写的结果 也是最好一次模型的结果
                    # ===============save best model=================
                    save_one_model(self.swin_model, self.optimizer, self.exp_name_dir, 'best_model')


            if self.opt.use_val_step:
                # ------------- print train info -----------
                print('Epoch:{:03d}\t train_loss:{:.4f}\t train_acc:{:.4f}\t lr:{:.5f} '
                      '-> best_epoch:{:03d}\t best_loss:{:.4f}\t best_acc:{:.4f}\t '
                      'best_OA:{:.4f}\t best_AA_mean:{:.4f}\t best_Kappa:{:.4f}\t'.format(epoch, train_loss, train_acc, lr,
                                                                                             best_epoch, best_loss, best_acc,
                                                                                             best_OA, best_AA_mean, best_Kappa))
            else:
                print('Epoch:{:03d}\t train_loss:{:.4f}\t train_acc:{:.4f}\t lr:{:.5f}'.format(epoch, train_loss, train_acc, lr))

            # ===============train log=================
            # 进入形如：log/exp_name_dir 的文件夹
            # 运行命令：tensorboard --logdir=train_log 启动tensorboard
            if self.opt.use_tensorboard:
                self.writer.add_scalar('lr/train', lr, self.opt.epochs)
                self.writer.add_scalar('loss/train', train_loss, self.opt.epochs)
                self.writer.add_scalar('acc/train', train_acc, self.opt.epochs)

            # save_one_model(self.swin_model, self.optimizer, self.exp_name_dir, 'epoch{:03d}-loss{:.4f}-acc{:.4f}'.format(epoch, train_loss, train_acc))
        # ===============save last model=================
        save_one_model(self.swin_model, self.optimizer, self.exp_name_dir, 'last_model')

        # ===============print best model info=================
        if self.opt.use_val_step:
            print("The best Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(best_epoch, best_loss, best_acc))
            show_result(best_epoch, best_OA, best_AA_mean, best_Kappa, best_AA, self.opt)


    @torch.no_grad()
    def valid_model(self):
        tar, pre = valid_one_epoch(self.swin_model, self.valid_dataloader, self.opt.device, self.exp_name_dir)

        OA, AA_mean, Kappa, AA = output_metric(tar, pre)
        # show_result(OA, AA_mean, Kappa, AA, self.opt)
        return OA, AA_mean, Kappa, AA


    @torch.no_grad()
    def predict_model(self):
        pos, pre = predict_one_epoch(self.swin_model, self.predict_dataloader, self.opt.device, self.exp_name_dir)
        # pos, pre = predict_one_epoch(self.swin_model, self.valid_dataloader, self.opt.device, self.exp_name_dir)
        # pos, pre = predict_one_epoch(self.swin_model, self.train_dataloader, self.opt.device, self.exp_name_dir)
        # ===============绘制类别预测图=================
        draw_data(pos, pre, self.opt.data_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('HSI')
    # 训练前模型配置
    parser.add_argument('--data_name', type=str, default='Houston', choices=['Indian', 'Pavia', 'Houston'], help='dataset to use')
    parser.add_argument('--train', type=bool, default=True, help='True means Train, False means Test')
    parser.add_argument('--seed', type=int, default=2, help='number of seed')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'], help='train device')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='dataloader batch_size')
    parser.add_argument('--swin_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='选择加载swin-transformer模型的类型')
    parser.add_argument('--process_data', type=str, default='batch', choices=['all', 'batch'], help='all 训练前一次性处理所有数据, batch 每一个batch处理数据一次')
    # 学习率调整
    parser.add_argument('--learn_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--scheduler_name', type=str, default='CosineAnnealingWarmRestarts', choices=['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'CosineAnnealingLR'], help='学习率衰减方式')
    parser.add_argument('--scheduler_gamma', type=float, default=0.9, help='StepLR参数 scheduler_gamma')       # StepLR参数
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
    parser.add_argument('--val_step', type=int, default=2, help='每隔多少epoch进行一次验证')
    # 是否启用early stopping
    parser.add_argument('--use_early_stopping', type=bool, default=False, help='是否使用早停')
    parser.add_argument('--early_stopping', type=int, default=3, help='连续多少epoch acc不发生相差小于0.5%就停止训练')
    # 数据处理方式
    parser.add_argument('--mode', type=str, default='band_pixel_patch', choices=['band_patch', 'pixel_patch', 'band_pixel_patch'], help='how the dataset is processed')  # 切记 如果为band_pixel_patch, 则参与运算的数据的batch_size会变为batch_size*band_patch
    parser.add_argument('--band_patch', type=int, default=2, help='band patch size,当mode为含有band时生效')
    parser.add_argument('--pixel_patch', type=int, default=9, help='the pixel patch size,当mode为含有pixel时生效')
    # 在band维进行数据降维
    parser.add_argument('--use_dimension_reduction', type=bool, default=False, help='是否使用数据降维')
    parser.add_argument('--dimension_mode', type=str, default='FA', choices=['FA', 'PCA'], help='因子分析和主成分分析')
    parser.add_argument('--numcomponents', type=int, default=3, help='降维后的特征数')
    # 使用 特征金字塔 和 残差结构
    parser.add_argument('--use_fp', type=bool, default=False, help='是否使用特征金字塔结构')
    parser.add_argument('--fp_name', type=str, default='FPN', choices=['SPP', 'PPM', 'ASPP', 'FPN', 'PANet'], help='可选的FP结构')
    parser.add_argument('--use_resblock', type=bool, default=True, help='是否使用残差结构')
    # 选择 损失函数
    parser.add_argument('--loss_function', type=str, default='CE_loss', choices=['CE_loss', 'focal_loss'], help='交叉熵损失 和 focal损失')
    parser.add_argument('--alpha_mode', type=str, default='handle', choices=['auto', 'handle', None], help='focal loss中alpha的获取方式,auto为自动根据训练数据类别比例获取,handle为人为根据数据类别比例手动划分,None为不划分alpha默认均为1')  # 当alpha=None时 focal loss效果和CE_loss一样
    # 使用通道注意力机制
    parser.add_argument('--use_cam', type=bool, default=False, help='通道注意力机制')
    # 使用 分支 空间 通道注意力机制
    parser.add_argument('--use_attention', type=bool, default=True, help='是否在模型开头使用空间或通道注意力机制')
    parser.add_argument('--change_channel', type=bool, default=False, help='是否在attention后改变通道数，在不使用fusion时使用')
    parser.add_argument('--use_fusion', type=bool, default=True, help='是否在使用attention后进行特征融合')
    opt = parser.parse_args()


    #  ['Indian', 'Pavia', 'Houston']
    train_and_test_model = Train_and_Test_Swin_model(opt)

    if opt.save_log:
        # 将训练过程中的print输出保存到result_dir文件夹中对应的txt文件中
        # 此时不会在 terminal中显示print输出
        output_file = save_log(train_and_test_model.result_dir)
        train_and_test_model.train_model()

        output_file.close()
    else:
        train_and_test_model.train_model()