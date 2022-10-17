# -*- coding: utf-8 -*-
# date: 2022/3/30
# Project: HSI-Classification
# File Name: read_result.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from scipy import io

scores = io.loadmat('./2DCNN-result.mat')
print(scores['OA'])