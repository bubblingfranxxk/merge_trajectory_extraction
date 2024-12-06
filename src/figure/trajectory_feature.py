# -*- coding = utf-8 -*-
# @Time : 2024/12/5 14:27
# @Author : 王砚轩
# @File : trajectory_feature.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取文件夹内的所有 CSV 文件并合并
folder_path = "/path/to/csv/folder"
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# 合并所有 CSV 文件
data_list = [pd.read_csv(file) for file in all_files]
combined_data = pd.concat(data_list, axis=0, ignore_index=True)

# 2. 针对每个特征列进行分析
features = combined_data.columns

# 创建输出文件夹
output_folder = "output_analysis"
os.makedirs(output_folder, exist_ok=True)

for feature in features:
    feature_data = combined_data[feature].dropna()  # 去除缺失值

    # 3. 计算均值和中位数
    mean_value = feature_data.mean()
    median_value = feature_data.median()
    print(f"Feature: {feature}, Mean: {mean_value:.2f}, Median: {median_value:.2f}")

    # 4. 绘制直方图和分布曲线
    plt.figure(figsize=(12, 6))
    sns.histplot(feature_data, bins=30, kde=True, color='blue', stat='density')
    plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='-', label=f'Median: {median_value:.2f}')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()

    # 保存图像
    plt.savefig(os.path.join(output_folder, f"{feature}_distribution.png"))
    plt.close()
