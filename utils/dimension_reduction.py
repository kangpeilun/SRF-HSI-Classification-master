# -*- coding: utf-8 -*-
# date: 2022/3/21 22:19
# Project: Swin-Transformer-New
# File Name: FA.py
# Description: 数据降维操作
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from factor_analyzer import FactorAnalyzer
import seaborn as sns


'''FA应用
    参考文章：https://mathpretty.com/10994.html
    
    df_cm = pd.DataFrame(np.abs(fa.loadings_), index=df.columns)
    plt.figure(figsize = (14,14))
    ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
    # 设置y轴的字体的大小
    ax.yaxis.set_tick_params(labelsize=15)
    plt.title('Factor Analysis', fontsize='xx-large')
    # Set y-axis label
    plt.ylabel('Sepal Width', fontsize='xx-large')
    plt.savefig('factorAnalysis.png', dpi=500)
'''
# ========================= 计算相关矩阵的特征值, 接着进行排序, 选择因子个数 =========================
def analysis_of_numcomponents(input_normalize):
    '''显示因子分解结果'''
    new_input_normalize = np.reshape(input_normalize, (-1, input_normalize.shape[2]))  # [145*145=21025, 200]
    # fa = FactorAnalysis(25, random_state=seed).fit_transform(new_input_normalize)
    fa = FactorAnalyzer(200, rotation=None).fit(new_input_normalize)
    # TODO: 了解如何生成特征矩阵
    ev, v = fa.get_eigenvalues()

    plt.scatter(range(1, new_input_normalize.shape[1] + 1), ev)
    plt.plot(range(1, new_input_normalize.shape[1] + 1), ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


def draw_fa_map(input_normalize):
    new_input_normalize = np.reshape(input_normalize, (-1, input_normalize.shape[2]))  # [145*145=21025, 200]
    fa = FactorAnalyzer(2, rotation='varimax').fit(new_input_normalize)
    df_cm = pd.DataFrame(np.abs(fa.loadings_))

    plt.figure(figsize=(50, 50))
    ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
    # 设置y轴的字体的大小
    ax.yaxis.set_tick_params(labelsize=15)
    plt.title('Factor Analysis', fontsize='xx-large')
    # Set y-axis label
    plt.ylabel('Sepal Width', fontsize='xx-large')
    plt.show()
    # plt.savefig('factorAnalysis.png', dpi=500)


def applyFA(input_normalize, numComponents=3):
    '''numComponents=3
        as the paper: SpectralNET: Exploring Spatial-Spectral WaveletCNN for Hyperspectral Image Classification
        exm: (145, 145, 200) -> (145, 145, numComponents)
    '''
    new_input_normalize = np.reshape(input_normalize, (-1, input_normalize.shape[2]))  # [145*145=21025, 200]
    # fa = FactorAnalysis(n_components=numComponents, random_state=seed)
    fa = FactorAnalyzer(numComponents, rotation=None).fit(new_input_normalize)
    # new_input_normalize = fa.fit_transform(new_input_normalize)  # [21025, 10]
    new_input_normalize = fa.transform(new_input_normalize)
    new_input_normalize = np.reshape(new_input_normalize, (input_normalize.shape[0], input_normalize.shape[1], numComponents))   # [145, 145, 10]
    return new_input_normalize


def applyPCA(input_normalize, numComponents=3):
    new_input_normalize = np.reshape(input_normalize, (-1, input_normalize.shape[2]))  # [145*145=21025, 200]
    pca = PCA(n_components=numComponents, whiten=True)
    new_input_normalize = pca.fit_transform(new_input_normalize)    # [21025, 10]
    new_input_normalize = np.reshape(new_input_normalize, (input_normalize.shape[0], input_normalize.shape[1], numComponents))   # [145, 145, 10]
    return new_input_normalize