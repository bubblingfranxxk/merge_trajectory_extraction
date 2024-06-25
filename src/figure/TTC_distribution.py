# -*- coding = utf-8 -*-
# @Time : 2024/6/5 22:55
# @Author : 王砚轩
# @File : TTC_distribution.py
# @Software: PyCharm
import os
import pandas as pd
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from TTC_acc_figure import create_output_folder


def main():
    rootPath = os.path.abspath('../../')
    tesetPath = rootPath + "/asset/Test/"
    file = pd.read_csv(tesetPath+"Pearson_Spearman_coefficient.csv")
    # 设置保存路径
    outputpath = tesetPath
    logger.info("Loading {}", tesetPath+"Pearson_Spearman_coefficient.csv")
    figurepath = tesetPath + "/distribution/"
    create_output_folder(tesetPath, "distribution")

    # 初始化存储DataFrame的列表
    dataColumns = ['Pearson1', 'Pearson2', 'Pearson3', 'Spearman1', 'Spearman2', 'Spearman3', 'DTW_Distance1',
                   'DTW_Distance2', 'DTW_Distance3']

    for col in dataColumns:
        temp = file[col]
        data = temp.where(temp != 999)
        # 筛选掉空白值和无穷值
        data = data[~np.isnan(data)]
        data = data[np.isfinite(data)]

        # 画图设置
        plt.figure(dpi=150)
        sns.set(style='whitegrid')
        sns.set_style("whitegrid", {"axes.facecolor": "#e9f3ea"})

        # 绘制分布图
        sns.histplot(data, kde=True)

        # 计算中位数、平均值、方差和峰度偏度
        median = np.nanmedian(data)
        mean = np.mean(data)
        variance = np.var(data)
        skewness = skew(data)
        kurt = kurtosis(data)

        # 标注中位数和平均值
        plt.axvline(x=median, color='r', linestyle='--', label=f'Median: {median:.2f}')
        plt.axvline(x=mean, color='g', linestyle='--', label=f'Mean: {mean:.2f}')

        # 显示方差和峰度偏度
        plt.text(2, 200, f'Variance: {variance:.2f}', fontsize=10)
        plt.text(2, 180, f'Skewness: {skewness:.2f}', fontsize=10)
        plt.text(2, 160, f'Kurtosis: {kurt:.2f}', fontsize=10)

        plt.legend()
        plt.savefig(figurepath+f"{col}.png")


if __name__ == '__main__':
    main()
