# -*- coding = utf-8 -*-
# @Time : 2024/12/5 14:27
# @Author : 王砚轩
# @File : trajectory_feature.py
# @Software: PyCharm


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def analyze_feature_distributions(feature_columns, data_path, output_folder):
    """
    分析多个CSV文件中指定特征列的分布，生成直方图和分布曲线，并输出均值和中位数。

    Args:
        feature_columns (list of str): 待分析的特征列名称列表。
        data_path (str): 包含CSV文件的文件夹路径。
        output_folder (str): 存储分析结果图像的文件夹路径。

    Returns:
        None
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取所有CSV文件并合并
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    data_list = [pd.read_csv(file) for file in all_files]
    combined_data = pd.concat(data_list, axis=0, ignore_index=True)

    for feature in feature_columns:
        if feature not in combined_data.columns:
            logger.error(f"Feature '{feature}' not found in data. Skipping...")
            continue

        feature_data = combined_data[feature].dropna()  # 去除缺失值

        # 计算均值和中位数
        mean_value = feature_data.mean()
        median_value = feature_data.median()
        logger.info(f"Feature: {feature}, Mean: {mean_value:.2f}, Median: {median_value:.2f}")

        # 绘制直方图和分布曲线
        plt.figure(figsize=(12, 6))
        sns.histplot(feature_data, bins=50, kde=True, color='blue', stat='density')
        plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
        plt.axvline(median_value, color='green', linestyle='-', label=f'Median: {median_value:.2f}')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()

        # 保存图像
        output_path = os.path.join(output_folder, f"{feature}_distribution.png")
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Distribution plot saved for feature '{feature}' at {output_path}")

if __name__ == '__main__':
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    traj_path = assetPath + "/extracted_data/"
    output_folder = assetPath + "/output_analysis/"
    features = ['lonVelocity', 'lonAcceleration', 'latAcceleration']

    analyze_feature_distributions(features, traj_path, output_folder)