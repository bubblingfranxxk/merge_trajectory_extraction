# -*- coding = utf-8 -*-
# @Time : 2025/3/31 23:22
# @Author : 王砚轩
# @File : traj_plot.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from loguru import logger
from matplotlib.ticker import MaxNLocator  # 新增

# 设置Seaborn的样式（可选）
sns.set(style="whitegrid")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

rootPath = os.path.abspath('../../')
assetPath = os.path.join(rootPath, 'asset')
processPath = os.path.join(assetPath, 'processed_data')
# 获取所有CSV文件
csv_files = [f for f in os.listdir(processPath) if f.endswith('.csv')]

# 定义需要绘制的字段对
pairs = [
    ("ego_lonLaneletPos", "ego_latLaneCenterOffset"),
    ("lead_lonLaneletPos", "lead_latLaneCenterOffset"),
    ("rear_lonLaneletPos", "rear_latLaneCenterOffset")
]

# 遍历所有 CSV 文件并进行绘图
for file_name in csv_files:
    file_path = os.path.join(processPath, file_name)
    df = pd.read_csv(file_path)

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("tab10", n_colors=len(pairs))

    plotted = False

    for idx, (col_lon, col_lat) in enumerate(pairs):
        # 选择对应的 mask 列
        mask = ("ego_mask" if col_lon.startswith("ego_")
                else "lead_mask" if col_lon.startswith("lead_")
        else "rear_mask")

        # 只保留掩码为 1 的行
        dfv = df[df[mask] == 1]

        if dfv.empty:
            continue

        # 再次过滤掉 999（如果需要）
        dfv = dfv[(dfv[col_lon] != 999) & (dfv[col_lat] != 999)]
        if dfv.empty:
            logger.warning(f"{file_name}: 在掩码过滤后，{col_lon}/{col_lat} 存在 999 或无有效点，跳过")
            continue

        # 绘制轨迹
        sns.lineplot(
            x=dfv[col_lon], y=dfv[col_lat],
            label=f"{col_lon.split('_')[0]}_trajectory",
            color=colors[idx], linewidth=3,
            ax=ax
        )
        plotted = True

    if not plotted:
        logger.error(f"{file_name}: 没有任何有效轨迹可绘制，跳过整张图")
        plt.close()
        continue
        # 获取当前坐标轴

    # 2. 固定坐标轴在 Figure 中的位置，给上下左右留足边距
    #    [left, bottom, right, top] 分别对应 figure 宽度的比例
    fig.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.18)

    # 1. 减少刻度数量：最多显示 6 个主刻度
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # 2. 调大刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=24)

    # 添加横纵轴标签和标题
    ax.set_xlabel('lonPos', fontsize=24)
    ax.set_ylabel('latPos', fontsize=24)
    ax.set_title(f"{file_name} 轨迹", fontsize=24)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(pairs), frameon=False, fontsize=24)
    ax.grid(False)

    # 保存高清图片
    base, _ = os.path.splitext(file_name)
    save_path = os.path.join(processPath, f"{base}.png")
    fig.savefig(
        save_path,
        dpi=500,
        bbox_inches='tight',
        transparent=False
    )
    plt.close(fig)
    logger.info(f"已保存图像: {save_path}")