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
from scipy.stats import f, skew, kurtosis
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
    # dataColumns = ['Pearson1', 'Pearson2', 'Pearson3', 'Spearman1', 'Spearman2', 'Spearman3', 'DTW_Distance1',
    #                'DTW_Distance2', 'DTW_Distance3']
    dataColumns = ['DTW_Distance1', 'DTW_Distance2', 'DTW_Distance3']

    for col in dataColumns:

        # # DTW 数据预处理 设置一个惩罚值
        # if "DTW" in col:
        #     dtw_temp = file[['DTW_Distance1', 'DTW_Distance2', 'DTW_Distance3']]
        #     penalty_value = dtw_temp.apply(lambda row: row.apply(lambda x: row.max() if x == 0 else x), axis=1)
        #     temp = penalty_value[col]
        #     data = temp.where(temp != 999)
        # else:
        #     temp = file[col]
        #     data = temp.where(temp != 999)

        temp = file[col]
        data = temp.where((temp > 0) & (temp < 999))

        # 筛选掉空白值和无穷值
        data = data[~np.isnan(data)]
        data = data[np.isfinite(data)]

        # 画图设置
        plt.figure(dpi=1500)
        sns.set(style='whitegrid')
        sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})

        # # 绘制分布图
        # sns.histplot(data, kde=True)

        # 使用seaborn绘制直方图
        plt.figure(figsize=(10, 6))
        if '1' in col:
            ax = sns.histplot(data, bins=500, kde=False, edgecolor=None, color='#5470C6')
        else:
            ax = sns.histplot(data, bins=1000, kde=False, edgecolor=None, color='#5470C6')

        # 获取y轴的最大值
        y_max = ax.get_ylim()[1]

        # 计算中位数、平均值、方差和峰度偏度
        median = np.nanmedian(data)
        mean = np.mean(data)
        variance = np.var(data)
        skewness = skew(data)
        kurt = kurtosis(data)

        # 标注中位数和平均值
        plt.axvline(x=median, color='#AAD795', linestyle='--', linewidth=4, label=f'Median: {median:.2f}')
        plt.axvline(x=mean, color='#FAC858', linestyle='--', linewidth=4, label=f'Mean: {mean:.2f}')

        # 定义文本位置
        y_position = y_max * 0.9  # 初始位置，设为y轴最大值的90%
        y_step = y_max * 0.1  # 每行文本之间的距离，设为y轴最大值的5%

        # 显示方差、偏度和峰度
        plt.text(15, y_position, f'Variance: {variance:.2f}', fontsize=20)
        plt.text(15, y_position - y_step, f'Skewness: {skewness:.2f}', fontsize=20)
        plt.text(15, y_position - 2 * y_step, f'Kurtosis: {kurt:.2f}', fontsize=20)

        ax.set_title('Histogram with Mean and Median', fontsize=20)
        ax.set_xlabel('Value', fontsize=20)
        ax.set_ylabel('Frequency', fontsize=20)

        plt.legend(fontsize=20)
        # 调整刻度标签的字号
        ax.tick_params(axis='both', which='major', labelsize=20)
        # 设置x轴范围为0到50
        if '1' in col:
            ax.set_xlim(0, 100)
        else:
            ax.set_xlim(0, 50)
        plt.savefig(figurepath+f"{col}.png")

    logger.info("Finish.")


if __name__ == '__main__':
    main()
