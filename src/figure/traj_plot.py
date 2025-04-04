# -*- coding = utf-8 -*-
# @Time : 2025/3/31 23:22
# @Author : 王砚轩
# @File : traj_plot.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# 设置Seaborn的样式（可选）
sns.set(style="whitegrid")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

rootPath = os.path.abspath('../../')
assetPath = rootPath + '/asset/'
processPath = assetPath + "/processed_data/"
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
    # 筛选出不包含999值的字段对
    valid_pairs = []
    for col_lon, col_lat in pairs:
        if (df[col_lon] == 999).any() or (df[col_lat] == 999).any():
            print(f"跳过绘制 {col_lon} 与 {col_lat}，存在999值")
        else:
            valid_pairs.append((col_lon, col_lat))

    # 如果没有有效的字段对则退出
    if not valid_pairs:
        print("所有选定的字段均存在999值，无有效数据进行绘图。")
        exit()
    # 创建图形
    plt.figure(figsize=(12, 6))

    # 使用tab10调色板
    colors = sns.color_palette("tab10", n_colors=len(df.columns))

    # 为每一组有效字段绘制粗虚线轨迹图
    for idx, (col_lon, col_lat) in enumerate(valid_pairs):
        sns.lineplot(x=df[col_lon], y=df[col_lat],
                     label=f"{col_lon} vs {col_lat}",
                     color=colors[idx],
                     linewidth=3)       # 线宽较粗

    # 添加横纵轴标签和标题
    plt.xlabel('Lanelet Position (lonPos)')
    plt.ylabel('Lane Center Offset (latPos)')
    plt.title('轨迹示例')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=16)  # 显示图例
    plt.grid(True)

    # 自动调整子图布局
    plt.tight_layout()

    file_name_list = file_name.split('_')
    # 保存高清图片（替换为你的保存路径）
    save_path = processPath+f'{file_name_list[0]}_{file_name_list[2]}.png'  # 支持.jpg/.pdf/.svg等格式
    plt.savefig(
        save_path,
        dpi=500,                 # 设置分辨率
        bbox_inches='tight',     # 自动裁剪空白区域
        transparent=False        # 背景透明可选
    )

    # 显示图形
    plt.close()