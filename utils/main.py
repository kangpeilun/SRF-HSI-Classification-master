# -*- coding: utf-8 -*-
# date: 2022/2/20 14:16
# Project: 毕业设计
# File Name: main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch import optim, nn

import os
from os.path import join
import numpy as np
from sklearn.metrics import confusion_matrix

from model.loss import focal_loss


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        # TODO: 思考当np.sum(matrix[i, :])=0时，将分母处理为1e-5是否合理
        AA[i] = matrix[i, i] / np.sum(matrix[i, :]) if np.sum(matrix[i, :])!=0 else 1e-5
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

# 用于计算平均准确率
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def train_one_epoch(swin_model, train_dataloader, device, criterion, optimizer):
    '''
    用于训练一个epoch的函数
    :param swin_model: 训练的模型
    :param train_dataloader: 训练集dataloader
    :param device: 训练设备
    :param criterion: 损失函数
    :param optimizer: 优化器
    :return: objs.avg(=train_loss), top1.avg(=train_acc), tar(=label), pre(=predict)
    '''
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    swin_model.train()
    for index, (x_train, y_train, data_info) in enumerate(train_dataloader):
        # x_train: [batch_size, c, h, w]  x_train_1 type: torch.float32   x_train_2 type:
        # data_info['pos'] (tensor) [[], [], ...]
        x_train, y_train = x_train.to(device), y_train.to(device)
        # ===============forward=================
        y_pre = swin_model(x_train, data_info)
        loss = criterion(y_pre, y_train)  # 注意训练模型的时候使用的是 模型直接预测的输出，和label计算loss，不需要根据索引将y_pre变为真实值

        prec1, t, p = accuracy(y_pre, y_train, topk=(1,))   # p (tensor) [class_id1, class_id2, ...]
        n = x_train.size(0)
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)

        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

        # ===============backward=================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return objs.avg, top1.avg, tar, pre


def valid_one_epoch(swin_model, valid_dataloader, device, save_dir_path):
    '''band_patch
    用于训练过程中的 验证部分, band_patch
    :param swin_model: 训练的模型
    :param valid_dataloader: 验证数据集
    :param device: 训练设备
    :param criterion: 损失函数
    :param save_dir_path: 模型保存的位置
    :return: tar(=label), pre(=predict)
    '''
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    # 加载模型
    # swin_model.load_state_dict(torch.load(f'./log/{save_dir_path}/best_model.pth'))
    swin_model.eval()
    for batch_idx, (x_test, y_test, data_info) in enumerate(valid_dataloader):
        x_test, y_test = x_test.to(device), y_test.to(device)
        # predict
        y_pre = swin_model(x_test, data_info)

        # t表示 target(标签)  p表示 predict(预测结果)
        prec1, t, p = accuracy(y_pre, y_test, topk=(1,))
        n = x_test.shape[0]
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


def predict_one_epoch(swin_model, predict_dataloader, device, save_dir_path):
    '''
    用于训练过程中的 验证部分
    :param swin_model: 训练的模型
    :param valid_dataloader: 测试数据集
    :param device: 训练设备
    :param criterion: 损失函数
    :param save_dir_path: 模型保存的位置
    :return: tar(=label), pre(=predict)
    '''
    pre = []
    pos = []
    # 加载模型
    load_one_model(swin_model, save_dir_path, 'best_model')
    swin_model.eval()
    for batch_idx, (x_test, y_test, data_info) in enumerate(predict_dataloader):
        x_test, y_test = x_test.to(device), y_test.to(device)
        # predict
        y_pre = swin_model(x_test, data_info)

        # t表示 target(标签)  p表示 predict(预测结果)
        prec1, t, p = accuracy(y_pre, y_test, topk=(1,))

        # pos_np = np.append(pos_np, data_info['pos'].numpy())  # 保存像素的坐标位置
        # pre_np = np.append(pre_np, p.cpu().numpy())     # 保存像素 预测结果

        pos.extend(list(data_info['pos'].numpy()))
        pre.extend(list(p.cpu().numpy()))

    return pos, pre


def write_txt(best_epoch, OA, AA_mean, Kappa, AA, opt, result_dir):
    opt = vars(opt)
    with open(join(result_dir, 'result.txt'), 'w', encoding='utf-8') as f:
        f.write("*"*10+"Final result Use the best model"+"*"*10+"\n")
        f.write('best_epoch'.rjust(20)+': '+str(best_epoch)+'\n')
        f.write('OA'.rjust(20)+': '+str(OA)+'\n')
        f.write('AA_mean'.rjust(20)+': '+str(AA_mean)+'\n')
        f.write('Kappa'.rjust(20)+': '+str(Kappa)+'\n')
        f.write('AA'.rjust(20)+': '+str(AA)+'\n')
        f.write('\n'+'-'*20+' Parameter '+'-'*20+'\n')
        for k, v in opt.items():
            f.write(k.rjust(20)+': '+str(v)+'\n')


def show_result(epochs, OA, AA_mean, Kappa, AA, opt):
    opt = vars(opt)
    print("*"*10, "Final result Use the best model", "*"*10)
    print("Epoch:{:03d}".format(epochs))
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))
    print(AA)
    print("**************************************************\n")
    print("Parameter:")
    for k, v in zip(opt.keys(), opt.values()):
        print(k.rjust(20)+': '+str(v))


def save_one_model(model, optimizer, save_path, model_name):
    '''保存训练过程中的一个模型'''
    torch.save(model.state_dict(), join(save_path, f'{model_name}.pth'))
    # torch.save(optimizer.state_dict(), join(save_path, f'{model_name}_optimizer.pth'))


def load_one_model(model, save_path, model_name):
    '''加载一个模型，用于测试模型'''
    print(join(save_path, f'{model_name}.pth'))
    model.load_state_dict(torch.load(join(save_path, f'{model_name}.pth'), map_location='cuda:0'))


def get_scheduler(opt, optimizer):
    '''获取学习率衰减方式'''
    if opt.scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_step_size, gamma=opt.scheduler_gamma)
    elif opt.scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    elif opt.scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.cawr_t_0, T_mult=opt.cawr_t_mult)   # T_0=10 每10个epoch 学习率 回复到初始值
    elif opt.scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)   # 让学习率余弦周期性变化5次

    return scheduler


def get_loss(opt, num_classes, alpha):
    if opt.loss_function == 'CE_loss':
        criterion = nn.CrossEntropyLoss()
    elif opt.loss_function == 'focal_loss':
        # criterion = focal_loss(num_classes=num_classes)
        criterion = focal_loss(alpha=alpha)

    return criterion