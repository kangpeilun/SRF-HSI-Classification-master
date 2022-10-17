# -*- coding: utf-8 -*-
# date: 2022/5/2
# Project: HSI-Classification
# File Name: draw_predict_data.py
# Description: 绘制预测结果的颜色
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageColor, ImageDraw
from .utils import check_dir
import os

def colors_per_class(data='Indian'):
    '''
    保存不同数据集每种类别的颜色
    :param data: ['Indian', 'Pavia', 'Houston']
    :return:
    '''
    if data == 'Indian':
        colors = {
            # 0: ['others', '#000000'],
            0: ['CornNotill', '#4fab49'],
            1: ['CornMintill', '#88ba43'],
            2: ['Corn', '#3e845c'],
            3: ['GrassPature', '#388445'],
            4: ['GrassTrees', '#915236'],
            5: ['HayWindrow', '#67bcc8'],
            6: ['SoybeanNotill', '#ffffff'],
            7: ['SoybeanMintill', '#c7b0ca'],
            8: ['SoybeanClean', '#db312d'],
            9: ['Wheat', '#782424'],
            10: ['Woods', '#2c6ab4'],
            11: ['BuilGraTrDri', '#e0dd54'],
            12: ['StoSteTower', '#d98e35'],
            13: ['Alfalfa', '#54307d'],
            14: ['GrassPastMow', '#e4775a'],
            15: ['Oats', '#9d5796'],
        }
    elif data == 'Pavia':
        colors = {
            # 0: ['others', '#000000'],
            0: ['Asphalt', '#c9cacc'],
            1: ['Meadows', '#6db146'],
            2: ['Gravel', '#67bcc8'],
            3: ['Trees', '#387b42'],
            4: ['Metal Sheets', '#9d5796'],
            5: ['Bare Soil', '#955333'],
            6: ['Bitumen', '#742d78'],
            7: ['Bricks', '#db312d'],
            8: ['Shadows', '#e0dd54'],
        }
    elif data == 'Houston':
        colors = {
            # 0: ['others', '#000000'],
            0: ['Healthy Grass', '#4fac49'],
            1: ['Stressed Grass', '#89bb43'],
            2: ['Synthetic Grass', '#3d855c'],
            3: ['Trees', '#388545'],
            4: ['Soil', '#915136'],
            5: ['Water', '#67bdc9'],
            6: ['Residential', '#ffffff'],
            7: ['Commercial', '#c8b1cb'],
            8: ['Road', '#dc2f2d'],
            9: ['Highway', '#792323'],
            10: ['Railway', '#3565a7'],
            11: ['Parkinig Lot1', '#e1dd54'],
            12: ['Parkinig Lot2', '#db8e35'],
            13: ['Tennis Court', '#55307d'],
            14: ['Running Track', '#e5775a'],
        }

    return colors


def create_image(data='Indian'):
    '''根据不同数据集创建不同尺寸的图像'''
    if data == 'Indian':
        w, h = 145, 145     # 图像尺寸
    elif data == 'Pavia':
        w, h = 340, 610
    elif data == 'Houston':
        w, h = 1905, 349

    image = Image.new('RGB', (w, h), (0,0,0))     # 创建黑色背景图
    return image


def draw_one_pixel(draw, xy, fill):
    draw.point(xy=xy, fill=fill)


def draw_data(positions, predicts, data='Indian'):
    '''
    绘制不同数据集的预测类别颜色图
    :param predictions:
    :param data:
    :return:
    '''
    colors = colors_per_class(data)     # 获取颜色表
    image = create_image(data)          # 创建黑色背景颜色图
    draw = ImageDraw.Draw(image)
    '''
    ImageColor模块使用方法见以下连接
    https://www.jianshu.com/p/f63dc8b5445d?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation
    '''
    with ThreadPoolExecutor(1000) as t:
        for pos, pre in zip(positions, predicts):
            # draw.point(list(pos), fill=ImageColor.getrgb(colors[pre][1]))     # 在对应索引位置处的像素上色
            xy = (pos[1], pos[0])
            fill = ImageColor.getrgb(colors[pre][1])
            t.submit(draw_one_pixel, draw=draw, xy=xy, fill=fill)

    check_dir(f'predict')
    image.save(os.path.join(f'predict/{data}.png'))
    # image.show(title=f"{data}'s predict result")


if __name__ == '__main__':
    pos = [[1,3], [4,5], [6,6]]
    pre = [1,3,5]
    draw_data(pos, pre, data='Indian')