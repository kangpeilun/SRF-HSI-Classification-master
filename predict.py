# -*- coding: utf-8 -*-
# date: 2022/5/2
# Project: HSI-Classification
# File Name: predict.py
# Description: 预测函数
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from main import Train_and_Test_Swin_model
from options.train_options import train_options, load_settings, show_options


config_file = [
    'predictions/indian/band-pixel-patch-5-4.yaml',
    'predictions/houston/band-patch-3-seed2.yaml',
    'predictions/pavia/band-pixel-patch-3-3-focal.yaml',
]


def predict():
    for file in config_file:
        settings = load_settings(file)
        opt = train_options(settings)
        train_and_test_model = Train_and_Test_Swin_model(opt)

        print('\n'+'-'*20, file, '-'*20)
        show_options(opt)
        train_and_test_model.predict_model()


if __name__ == '__main__':
    predict()