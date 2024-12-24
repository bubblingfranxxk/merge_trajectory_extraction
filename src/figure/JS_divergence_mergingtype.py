# -*- coding = utf-8 -*-
# @Time : 2024/12/15 17:16
# @Author : 王砚轩
# @File : JS_divergence_mergingtype.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


def calculate_js_divergence(data, feature_columns, merging_type_col="MergingType"):
    # 获取所有MergingType分类
    merging_types = sorted(data[merging_type_col].unique())  # 确保 A-H 顺序
    js_results = {col: np.zeros((len(merging_types), len(merging_types))) for col in feature_columns}

    # 对每个特征列，计算类别分布并计算JS散度
    for col in feature_columns:
        distributions = {}

        # 计算每个类别的分布
        for mt in merging_types:
            feature_data = data[data[merging_type_col] == mt][col]
            hist, _ = np.histogram(feature_data, bins=30, density=True)  # 30个bins，归一化
            distributions[mt] = hist + 1e-10  # 防止log(0)

        # 计算两两JS散度
        for i, j in combinations(range(len(merging_types)), 2):
            mt1, mt2 = merging_types[i], merging_types[j]
            js_div = jensenshannon(distributions[mt1], distributions[mt2])
            js_results[col][i, j] = js_results[col][j, i] = js_div

    return js_results, merging_types


# 保存JS散度矩阵为CSV文件
def save_js_to_csv(js_results, merging_types, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for feature, matrix in js_results.items():
        df = pd.DataFrame(matrix, index=merging_types, columns=merging_types)
        csv_path = os.path.join(output_folder, f"JS_Divergence_{feature}.csv")
        df.to_csv(csv_path, index=True)
        logger.info(f"Saved JS divergence matrix for {feature} to {csv_path}")


# 绘制热力图
def plot_js_heatmaps(js_results, merging_types, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # 创建一个3x2的子图布局
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))  # 调整整体图像大小
    axes = axes.flatten()  # 将子图数组展平成1D列表，便于遍历

    for idx, (feature, matrix) in enumerate(js_results.items()):
        ax = axes[idx]
        sns.heatmap(
            matrix,
            xticklabels=merging_types,
            yticklabels=merging_types,
            annot=True,
            fmt=".4f",
            cmap="viridis",
            annot_kws={"size": 14},  # 调整单元格内文字大小
            ax=ax
        )
        ax.set_title(f"JS Divergence for {feature}", fontsize=18)  # 子图标题
        ax.set_xlabel("Merging Type", fontsize=14)
        ax.set_ylabel("Merging Type", fontsize=14)
        ax.tick_params(axis='x', labelsize=30)  # 调整X轴刻度文字大小
        ax.tick_params(axis='y', labelsize=30)  # 调整Y轴刻度文字大小

    # 删除多余的子图
    for idx in range(len(js_results), len(axes)):
        fig.delaxes(axes[idx])

    # 调整子图间的间距
    plt.tight_layout()

    # 保存整体图像
    combined_path = os.path.join(output_folder, "Combined_JS_Divergence_Heatmaps.png")
    plt.savefig(combined_path)
    plt.close()
    logger.info(f"Saved combined heatmaps to {combined_path}")


# 主函数
def main(folder_path, feature_columns, merging_type_col="MergingType", output_folder='output'):
    all_data = []

    # 遍历文件夹中的CSV文件，合并数据
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            all_data.append(pd.read_csv(file_path))

    data = pd.concat(all_data, ignore_index=True)

    # 计算JS散度
    js_results, merging_types = calculate_js_divergence(data, feature_columns, merging_type_col)

    # 保存CSV文件
    save_js_to_csv(js_results, merging_types, output_folder)

    # 绘制热力图
    plot_js_heatmaps(js_results, merging_types, output_folder)


if __name__ == '__main__':
    # 调用主函数
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    folder_path = assetPath + '/normalized_data/'
    feature_columns = ['traveledDistance', 'latLaneCenterOffset', 'heading', 'lonVelocity',
                      'lonAcceleration', 'latAcceleration']  # 替换为你的特征列名
    outputPath = assetPath + '/mergingType_JS/'
    main(folder_path, feature_columns, output_folder=outputPath)
