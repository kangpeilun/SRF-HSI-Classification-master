# -*- coding: utf-8 -*-
# date: 2022/3/28
# Project: HSI-Classification
# File Name: houston_config.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com


houston_config_file = [
    # # band-patch res
    # 'houston/band_patch_res/band-patch-1-res.yaml',
    # 'houston/band_patch_res/band-patch-2-res.yaml',
    # 'houston/band_patch_res/band-patch-3-res.yaml',
    # 'houston/band_patch_res/band-patch-4-res.yaml',
    # 'houston/band_patch_res/band-patch-5-res.yaml',
    # 'houston/band_patch_res/band-patch-6-res.yaml',
    # 'houston/band_patch_res/band-patch-7-res.yaml',
    # 'houston/band_patch_res/band-patch-8-res.yaml',
    # 'houston/band_patch_res/band-patch-9-res.yaml',
    # 'houston/band_patch_res/band-patch-10-res.yaml',

    # # pixel-patch res
    # 'houston/pixel_patch_res/pixel-patch-1-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-2-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-3-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-4-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-5-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-6-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-7-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-8-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-9-res-all-std.yaml',
    # 'houston/pixel_patch_res/pixel-patch-10-res-all-std.yaml',

    # # pixel-patch res focal
    # 'houston/pixel_patch_res_focal/pixel-patch-5-res-focal.yaml',

    # # band-patch 修改了bug，drop_last=Fasle， 现在第 15 16类可以学出来了
    # 'houston/band_patch_res_batch/band-patch-1-res-batch.yaml',
    # 'houston/band_patch_res_batch/band-patch-2-res-batch.yaml',
    # 'houston/band_patch_res_batch/band-patch-3-res-batch.yaml',         # band-patch=3 已经可以超过baseline
    # 'houston/band_patch_res_batch/band-patch-4-res-batch.yaml',
    # 'houston/band_patch_res_batch/band-patch-5-res-batch.yaml',
    # 'houston/band_patch_res_batch/band-patch-6-res-batch.yaml',
    # 'houston/band_patch_res_batch/band-patch-7-res-batch.yaml',
    # 'houston/band_patch_res_batch/band-patch-8-res-batch.yaml',
    # 'houston/band_patch_res_batch/band-patch-9-res-batch.yaml',
    # 'houston/band_patch_res_batch/band-patch-10-res-batch.yaml',

    # # pixel-patch 修改了bug，drop_last=Fasle， 现在第 15 16类可以学出来了
    # 'houston/pixel_patch_res_batch/pixel-patch-1-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-2-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-3-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-4-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-5-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-6-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-7-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-8-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-9-res-batch.yaml',
    # 'houston/pixel_patch_res_batch/pixel-patch-10-res-batch.yaml',

    # # band-pixel-patch
    # 'houston/band_pixel_patch_batch/band-pixel-patch-3-5-batch.yaml',

    # # 使用之前最好一组参数重新实验, 这次另val_step=1 效果可能会更好
    # 'houston/best_retrain/band-patch-3-seed0.yaml',       # seed=0 对于houston数据可能不是一组好的初始化 seed=2的效果更好
    'houston/best_retrain/band-patch-3-seed2.yaml',       # seed=0 对于houston数据可能不是一组好的初始化 seed=2的效果更好   val_step=2对与houston数据效果更好
]