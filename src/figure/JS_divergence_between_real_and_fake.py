# -*- coding = utf-8 -*-
# @Time : 2024/12/14 17:34
# @Author : 王砚轩
# @File : JS_divergence_between_real_and_fake.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from loguru  import logger

def compute_js_divergence(p, q):
    """计算两个概率分布 p 和 q 的 JS 散度"""
    p = np.asarray(p)
    q = np.asarray(q)
    # 平均分布
    m = 0.5 * (p + q)
    # 计算 JS 散度
    js_divergence = 0.5 * (entropy(p, m) + entropy(q, m))
    return js_divergence


def load_folder_data(folder_path, feature_columns):
    """加载文件夹中的所有 CSV 文件并提取指定的特征列"""
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            if set(feature_columns).issubset(df.columns):
                all_data.append(df[feature_columns])
            else:
                logger.debug(f"feature columns:{feature_columns}.")
                logger.debug(f"real data columns:{df.columns}")
                logger.warning(f"Warning: {file_path} does not contain all feature_columns.")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        logger.error("Error: No valid data found in the folder.")
        return pd.DataFrame()


def calculate_histogram(data, bins):
    """计算连续数据的归一化直方图作为概率分布"""
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    return hist, bin_edges


def calculate_js_divergences(real_file, fake_folder, feature_columns, output_file, bins=50):
    """计算 real_data 和每个 fake_data_epoch 文件在指定特征上的 JS 散度"""
    # 加载 real_data
    real_data = load_folder_data(real_file, feature_columns)
    if real_data.empty:
        logger.error("Error: real_data is empty or does not contain the specified feature_columns.")
        return

    # 计算真实数据的分布（直方图）
    real_distributions = {
        column: calculate_histogram(real_data[column], bins=bins)
        for column in feature_columns
    }

    js_results = []

    # 遍历 fake_data 文件夹中的每个 epoch 文件
    for file in os.listdir(fake_folder):
        if file.startswith("fake_data_epoch_") and file.endswith(".csv"):
            epoch = int(file.split("_")[-1].split(".")[0])
            fake_file_path = os.path.join(fake_folder, file)

            # 加载 fake_data
            fake_data = pd.read_csv(fake_file_path)
            if fake_data.empty or not set(feature_columns).issubset(fake_data.columns):
                logger.warning(f"Skipping {file} due to missing or empty feature_columns.")
                continue

            # 计算每个特征的 JS 散度
            js_for_epoch = {"epoch": epoch}
            for column in feature_columns:
                # 计算假数据的分布（直方图）
                fake_hist, _ = calculate_histogram(fake_data[column], bins=bins)

                # 获取真实数据的分布
                real_hist, bin_edges = real_distributions[column]

                # 对齐直方图（假设 bin_edges 一致）
                if len(real_hist) != len(fake_hist):
                    logger.warning(f"Warning: Bin mismatch for feature {column} at epoch {epoch}.")
                    continue

                # 计算 JS 散度
                js_divergence = compute_js_divergence(real_hist, fake_hist)
                js_for_epoch[column] = js_divergence

            js_results.append(js_for_epoch)

    # 将结果保存为 CSV 文件
    if js_results:
        result_df = pd.DataFrame(js_results)
        result_df.to_csv(output_file, index=False)
        logger.debug(f"JS divergence results saved to {output_file}")


if __name__ == '__main__':
    # 指定文件夹路径和特征列
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + '/asset/'
    real_folder = assetPath + '/normalized_data/'
    fake_folder = assetPath + '/GENERATED_DATA/'
    feature_columns = ['lonVelocity', 'lonAcceleration', 'latAcceleration']  # 替换为实际的特征列

    # 计算 JS 散度
    calculate_js_divergences(real_folder, fake_folder, feature_columns, assetPath+'JS_DIVERGENCE_RESULT.csv', bins=50)
