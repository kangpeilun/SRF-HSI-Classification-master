# -*- coding: utf-8 -*-
# date: 2022/3/28
# Project: HSI-Classification
# File Name: indian_config.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com


indian_config_file = [
    # # band-patch
    # 'indian/band_patch/band-patch-1.yaml',
    # 'indian/band_patch/band-patch-2.yaml',
    # 'indian/band_patch/band-patch-3.yaml',
    # 'indian/band_patch/band-patch-4.yaml',
    # 'indian/band_patch/band-patch-5.yaml',
    # 'indian/band_patch/band-patch-6.yaml',
    # 'indian/band_patch/band-patch-7.yaml',
    # 'indian/band_patch/band-patch-8.yaml',
    # 'indian/band_patch/band-patch-9.yaml',
    # 'indian/band_patch/band-patch-10.yaml',
    # 'indian/band_patch/band-patch-11.yaml',
    # 'indian/band_patch/band-patch-12.yaml',
    # 'indian/band_patch/band-patch-13.yaml',
    # 'indian/band_patch/band-patch-14.yaml',   # best  band-patch-14
    # 'indian/band_patch/band-patch-15.yaml',
    # 'indian/band_patch/band-patch-16.yaml',
    # 'indian/band_patch/band-patch-17.yaml',
    # 'indian/band_patch/band-patch-18.yaml',
    # 'indian/band_patch/band-patch-19.yaml',
    # 'indian/band_patch/band-patch-20.yaml',
    # # pixel_patch
    # 'indian/pixel_patch/pixel-patch-1.yaml',
    # 'indian/pixel_patch/pixel-patch-2.yaml',
    # 'indian/pixel_patch/pixel-patch-3.yaml',
    # 'indian/pixel_patch/pixel-patch-4.yaml',  # best pixel-patch-4
    # 'indian/pixel_patch/pixel-patch-5.yaml',
    # 'indian/pixel_patch/pixel-patch-6.yaml',
    # 'indian/pixel_patch/pixel-patch-7.yaml',
    # 'indian/pixel_patch/pixel-patch-8.yaml',
    # 'indian/pixel_patch/pixel-patch-9.yaml',
    # # band-pixel-patch
    # 'indian/band_pixel_patch/band-pixel-patch-1-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-2-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-3-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-4-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-5-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-6-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-7-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-8-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-9-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-10-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-11-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-12-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-13-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-14-1.yaml',
    # 'indian/band_pixel_patch/band-pixel-patch-15-1.yaml',
    # # 数据降维
    # 'indian/band_pixel_patch/band-pixel-patch-4-7.yaml',      # 效果不好
    # 'indian/band_pixel_patch/band-pixel-patch-4-7-dr3.yaml',  # 效果不好
    # # 使用 ResBlock
    # 'indian/band_patch_res/band-patch-1-res.yaml',
    # 'indian/band_patch_res/band-patch-2-res.yaml',
    # 'indian/band_patch_res/band-patch-3-res.yaml',
    # 'indian/band_patch_res/band-patch-4-res.yaml',
    # 'indian/band_patch_res/band-patch-5-res.yaml',
    # 'indian/band_patch_res/band-patch-6-res.yaml',
    # 'indian/band_patch_res/band-patch-7-res.yaml',
    # 'indian/band_patch_res/band-patch-8-res.yaml',
    # 'indian/band_patch_res/band-patch-9-res.yaml',
    # 'indian/band_patch_res/band-patch-10-res.yaml',
    # 'indian/band_patch_res/band-patch-11-res.yaml',
    # 'indian/band_patch_res/band-patch-12-res.yaml',
    # 'indian/band_patch_res/band-patch-13-res.yaml',
    # 'indian/band_patch_res/band-patch-14-res.yaml',
    # 'indian/band_patch_res/band-patch-15-res.yaml',
    # 'indian/band_patch_res/band-patch-16-res.yaml',
    # 'indian/band_patch_res/band-patch-17-res.yaml',
    # 'indian/band_patch_res/band-patch-18-res.yaml',
    # 'indian/band_patch_res/band-patch-19-res.yaml',
    # # 使用focal loss    效果不好
    # 'indian/band_patch_res_fcloss/band-patch-1-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-2-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-3-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-4-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-5-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-6-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-7-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-8-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-9-res-fcloss.yaml',
    # 'indian/band_patch_res_fcloss/band-patch-10-res-fcloss.yaml',
    # # 使用 swin 的 small base两个版本进行实验，对比之后 还是 tiny的效果最好
    # 'indian/band_patch_res_swintype/band-patch-4-res-small.yaml',
    # 'indian/band_patch_res_swintype/band-patch-4-res-base.yaml',
    # # 使用 band-pixel-patch res   效果没有 band-patch res 效果好
    # 'indian/band_pixel_patch_res/band-pixel-patch-4-1-res.yaml',
    # 'indian/band_pixel_patch_res/band-pixel-patch-4-2-res.yaml',
    # 'indian/band_pixel_patch_res/band-pixel-patch-4-3-res.yaml',
    # 'indian/band_pixel_patch_res/band-pixel-patch-4-4-res.yaml',
    # 'indian/band_pixel_patch_res/band-pixel-patch-4-5-res.yaml',
    # 'indian/band_pixel_patch_res/band-pixel-patch-4-6-res.yaml',
    # 'indian/band_pixel_patch_res/band-pixel-patch-4-7-res.yaml',
    # # 测试 all 和 batch 两种数据加载方式     all和batch的效果一致，只是all需要更多的epoch才能到达最好的效果
    # 'indian/band_patch_res_process/band-patch-4-res-all.yaml',
    # # 测试 swin-tiny 中 block的层级 2 4 6      三种不同层次的 basicblock个数，影响的是模型到达最好效果的epoch，block越小，epoch越小，实验证明 block=6的效果要略好一些
    # 'indian/band_patch_res_block/band-patch-4-res-block2.yaml',
    # 'indian/band_patch_res_block/band-patch-4-res-block4.yaml',
    # # 测试 swin-tiny 中 window_size 的大小 3 5 7   就结果而言，还是window_size=7的时候效果最好
    # 'indian/band_patch_res_window/band-patch-4-res-window3.yaml',
    # 'indian/band_patch_res_window/band-patch-4-res-window5.yaml',
    # # pixel-patch-4 对数据进行 标准化， 并使用 all加载方式            发现是归一化的锅，使用标准化 之后 直接就可以学到第15 16类了
    # 'indian/pixel_or_patch_res_all_std/pixel-patch-4-res-all-std.yaml',         # 标准化 和 归一化 比，标准化的效果比 归一化好多了
    # 'indian/pixel_or_patch_res_all_std/pixel-patch-4-res-all-normal.yaml',
    # 'indian/pixel_or_patch_res_all_std/band-patch-4-res-batch-std.yaml',        # 实验表明，band-patch 这种数据处理方法是有问题的 这样无法学到 第15 16类
    # 'indian/pixel_or_patch_res_all_std/pixel-patch-4-res-batch-std.yaml',       # 我发现是我 数据处理的有问题。可能并不是 归一化和标准化的锅          # 我自己的band-patch 和 pixel-patch代码有问题， 目前先使用pixel-patch all 2号最新代码

    # # pixel-patch 1到10        pixel-patch的效果很好，15 16类可以学出来了
    # 'indian/pixel_patch_res/pixel-patch-1-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-2-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-3-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-4-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-5-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-6-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-7-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-8-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-9-res-all-std.yaml',
    # 'indian/pixel_patch_res/pixel-patch-10-res-all-std.yaml',

    # # pixel-patch 使用特征金字塔试一试          发现还是 不使用 特征金字塔效果更好
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-fpn.yaml',        # 效果还没有只是用 pixel-patch 好
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-panet.yaml',        # 效果还没有只是用 pixel-patch 好, 比fpn还差
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-spp.yaml',            # 效果  fpn > spp > panet
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-ppm.yaml',            # 效果 ppm约等于 不使用 fp的效果
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-aspp.yaml',             # 效果 很差

    # # pixel-patch 使用dropout试一试            发现drop_out最终的效果比不用 稍微低一些，但是其可以更快的到达最好效果的epoch
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-drop0.1.yaml',        # 使用att_drop_rate可以更快的到达与不使用att_drop_rate相近的效果, 同时使用drop_rate和att_drop_rate效果很差
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-drop0.2.yaml',
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-drop0.3.yaml',
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-drop0.4.yaml',
    # 'indian/pixel_patch_fp_drop/pixel-patch-5-res-drop0.5.yaml',

    # # pixel-patch 使用focal loss, alpha根据预测的结果进行调整  batch 改为64试一试       # focal loss的效果并不稳定，且大概率会降低分类的精度
    # 'indian/pixel_patch_res_focal/pixel-patch-5-res-focal.yaml',
    # 'indian/pixel_patch_res_focal/pixel-patch-6-res-focal.yaml',

    # # band-patch 修改了bug，drop_last=Fasle， 现在第 15 16类可以学出来了     修改bug后效果非常好
    # 'indian/band_patch_res_batch/band-patch-1-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-2-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-3-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-4-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-5-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-6-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-7-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-8-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-9-res-batch.yaml',
    # 'indian/band_patch_res_batch/band-patch-10-res-batch.yaml',

    # # pixel-patch 修改了bug，drop_last=Fasle， 现在第 15 16类可以学出来了   修改bug后效果非常好
    # 'indian/pixel_patch_res_batch/pixel-patch-1-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-2-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-3-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-4-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-5-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-6-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-7-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-8-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-9-res-batch.yaml',
    # 'indian/pixel_patch_res_batch/pixel-patch-10-res-batch.yaml',

    # # 使用band-pixel-patch，选用单独使用band和pixel patch时最好一次的结果进行组合
    # 'indian/band_pixel_patch_batch/band-pixel-patch-5-4-batch.yaml',        # seed=2 该组合已经可以超过 baseline； seed=0已经完全超过baseline
    # 'indian/band_pixel_patch_batch/band-pixel-patch-6-4-batch.yaml',        # 效果没有 5-4 好
    # 'indian/band_pixel_patch_batch/band-pixel-patch-5-7-batch.yaml',        # 效果比 6-4 还差
    # 'indian/band_pixel_patch_batch/band-pixel-patch-5-5-batch.yaml',        # 效果与 5-4 低一些
    # 'indian/band_pixel_patch_batch/band-pixel-patch-5-3-batch.yaml',          # 效果不好

    # # 使用之前最好一次的结果，使用val_step=1重新进行训练，效果可能会变得更好
    # 'indian/best_retrain/band-pixel-patch-5-4.yaml',        # val_step=1后 能得到更好的模型了

    # # 使用添加了自适应注意力机制的模型    效果都很差
    # 'indian/band_pixel_patch_attention/band-pixel-patch-5-4-att.yaml',      # 效果 很差，没有原来的效果好
    # 'indian/band_pixel_patch_attention/band-pixel-patch-5-4-att-change.yaml',      # 效果 很差，没有原来的效果好
    # 'indian/band_pixel_patch_attention/band-pixel-patch-5-4-att-fusion.yaml',   # 进行特征融合

    # # best_retrain 测试训练结果是否保持不变
    # 'indian/best_retrain/band-pixel-patch-5-4.yaml',

    # # 补充 不使用 resblock 的实验
    # 'indian/fusion-exp/band-pixel-patch-5-4-nores.yaml',
    #
    # 'indian/fusion-exp/band-patch-1-nores.yaml',
    # 'indian/fusion-exp/band-patch-2-nores.yaml',
    # 'indian/fusion-exp/band-patch-3-nores.yaml',
    # 'indian/fusion-exp/band-patch-4-nores.yaml',
    # 'indian/fusion-exp/band-patch-5-nores.yaml',
    # 'indian/fusion-exp/band-patch-6-nores.yaml',
    # 'indian/fusion-exp/band-patch-7-nores.yaml',
    # 'indian/fusion-exp/band-patch-8-nores.yaml',
    # 'indian/fusion-exp/band-patch-9-nores.yaml',
    # 'indian/fusion-exp/band-patch-10-nores.yaml',

    'indian/fusion-exp/pixel-patch-1-nores.yaml',
    'indian/fusion-exp/pixel-patch-2-nores.yaml',
    'indian/fusion-exp/pixel-patch-3-nores.yaml',
    'indian/fusion-exp/pixel-patch-4-nores.yaml',
    'indian/fusion-exp/pixel-patch-5-nores.yaml',
    'indian/fusion-exp/pixel-patch-6-nores.yaml',
    'indian/fusion-exp/pixel-patch-7-nores.yaml',
    'indian/fusion-exp/pixel-patch-8-nores.yaml',
    'indian/fusion-exp/pixel-patch-9-nores.yaml',
    'indian/fusion-exp/pixel-patch-10-nores.yaml',
]