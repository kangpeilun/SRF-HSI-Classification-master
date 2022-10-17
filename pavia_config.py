# -*- coding: utf-8 -*-
# date: 2022/3/28
# Project: HSI-Classification
# File Name: pavia_config.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com


pavia_config_file = [
    # # band-patch res
    # 'pavia/band_patch_res/band-patch-1-res.yaml',
    # 'pavia/band_patch_res/band-patch-2-res.yaml',
    # 'pavia/band_patch_res/band-patch-3-res.yaml',
    # 'pavia/band_patch_res/band-patch-4-res.yaml',
    # 'pavia/band_patch_res/band-patch-5-res.yaml',
    # 'pavia/band_patch_res/band-patch-6-res.yaml',
    # 'pavia/band_patch_res/band-patch-7-res.yaml',
    # 'pavia/band_patch_res/band-patch-8-res.yaml',
    # 'pavia/band_patch_res/band-patch-9-res.yaml',
    # 'pavia/band_patch_res/band-patch-10-res.yaml',

    # # band-patch res
    # 'pavia/pixel_patch_res/pixel-patch-1-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-2-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-3-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-4-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-5-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-6-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-7-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-8-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-9-res-all-std.yaml',
    # 'pavia/pixel_patch_res/pixel-patch-10-res-all-std.yaml',

    # # pixel-patch res focal
    # 'pavia/pixel_patch_res_focal/pixel-patch-3-res-focal.yaml',
    # 'pavia/pixel_patch_res_focal/pixel-patch-5-res-focal.yaml',

    # # band-patch 修改了bug，drop_last=Fasle， 现在第 15 16类可以学出来了
    # 'pavia/band_patch_res_batch/band-patch-1-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-2-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-3-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-4-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-5-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-6-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-7-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-8-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-9-res-batch.yaml',
    # 'pavia/band_patch_res_batch/band-patch-10-res-batch.yaml',

    # # pixel-patch 修改了bug，drop_last=Fasle， 现在第 15 16类可以学出来了
    # 'pavia/pixel_patch_res_batch/pixel-patch-1-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-2-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-3-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-4-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-5-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-6-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-7-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-8-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-9-res-batch.yaml',
    # 'pavia/pixel_patch_res_batch/pixel-patch-10-res-batch.yaml',

    # # 使用band-pixel-patch，选用单独使用band和pixel patch时最好一次的结果进行组合
    # 'pavia/band_pixel_patch_batch/band-pixel-patch-3-2-batch.yaml',       # seed=2 效果比单独用band和pixel patch 好许多
    # 'pavia/band_pixel_patch_batch/band-pixel-patch-3-6-batch.yaml',
    # 'pavia/band_pixel_patch_batch/band-pixel-patch-3-3-batch.yaml',         # 3-3的效果比3-2好一丢丢
    # 'pavia/band_pixel_patch_batch/band-pixel-patch-3-4-batch.yaml',

    # # 使用band-pixel-patch，选用seed=0，1，2分别试一试
    # 'pavia/band_pixel_patch_batch/seed0/band-pixel-patch-3-2-batch.yaml',   # seed=0
    # 'pavia/band_pixel_patch_batch/seed0/band-pixel-patch-3-3-batch.yaml',   # seed=0 3-3 的效果最好，与baseline差距在1%之内
    # 'pavia/band_pixel_patch_batch/seed0/band-pixel-patch-3-4-batch.yaml',
    # 'pavia/band_pixel_patch_batch/seed0/band-pixel-patch-3-5-batch.yaml',

    # 'pavia/band_pixel_patch_batch/seed1/band-pixel-patch-3-2-batch.yaml',   # seed=1
    # 'pavia/band_pixel_patch_batch/seed1/band-pixel-patch-3-3-batch.yaml',
    # 'pavia/band_pixel_patch_batch/seed1/band-pixel-patch-3-4-batch.yaml',
    # 'pavia/band_pixel_patch_batch/seed1/band-pixel-patch-3-5-batch.yaml',
    #
    # 'pavia/band_pixel_patch_batch/seed2/band-pixel-patch-3-2-batch.yaml',   # seed=2
    # 'pavia/band_pixel_patch_batch/seed2/band-pixel-patch-3-3-batch.yaml',
    # 'pavia/band_pixel_patch_batch/seed2/band-pixel-patch-3-4-batch.yaml',
    # 'pavia/band_pixel_patch_batch/seed2/band-pixel-patch-3-5-batch.yaml',

    # # 使用通道注意力机制  效果不好
    # 'pavia/band_pixel_patch_cam/band-pixel-patch-3-3-batch.yaml',

    # # 使用focal loss, 以及不同的随机种子， val_step=1 精细测试
    # 'pavia/band_pixel_patch_seed/band-pixel-patch-3-3-seed0-focal.yaml',    # 发现val_step=1 对结果会有极大的影响

    # # 使用之前最好一组参数重新实验, 这次另val_step=1 效果可能会更好
    # 'pavia/best_retrain/band-pixel-patch-3-3.yaml',
    # 'pavia/best_retrain/band-pixel-patch-3-3-focal.yaml',       # focal loss 的效果非常好
]