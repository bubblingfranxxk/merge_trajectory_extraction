# -*- coding = utf-8 -*-
# @Time : 2024/7/3 9:35
# @Author : 王砚轩
# @File : danger_extraction.py
# @Software: PyCharm
import os
import pandas as pd
import matplotlib.pyplot as plt


def check_and_save_csv(input_folder, output_folder, figure_folder, columns_to_check, value_range):
    # 创建输出文件夹和figure文件夹
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(figure_folder, exist_ok=True)

    # 获取所有csv文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # 用于存储满足条件的文件
    valid_files = []

    for csv_file in csv_files:
        if "others" in csv_file:
            continue
        file_path = os.path.join(input_folder, csv_file)
        df = pd.read_csv(file_path)

        # 检查指定列是否有值在特定范围内
        condition_met = df[columns_to_check].apply(lambda x: (x > value_range[0]) & (x < value_range[1])).any().any()

        if condition_met:
            # 保存csv文件到新的文件夹
            new_file_name = 'danger_' + csv_file
            new_file_path = os.path.join(output_folder, new_file_name)
            df.to_csv(new_file_path, index=False)
            valid_files.append((new_file_name, df))

    # 绘制折线图，每张图包含3个CSV文件，每个CSV文件5个子图
    num_plots_per_figure = 3
    num_columns_per_plot = 5
    for i in range(0, len(valid_files), num_plots_per_figure):
        fig, axes = plt.subplots(num_plots_per_figure, num_columns_per_plot, figsize=(15, 9))
        fig.suptitle(f'Danger Data from CSV Files {i + 1} to {i + num_plots_per_figure}')

        for j, (csv_file, df) in enumerate(valid_files[i:i + num_plots_per_figure]):
            for k, column in enumerate(columns_to_check):
                ax = axes[j, k]
                ax.plot(df[column], label=column)
                ax.set_title(f'{csv_file} - {column}')
                ax.legend()
                ax.set_ylim([df[column].min()-10, df[column].max()+10])  # 根据需要调整Y轴范围

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以适应suptitle
        figure_file_name = os.path.join(figure_folder, f'danger_plot_{i // num_plots_per_figure + 1}.png')
        plt.savefig(figure_file_name)
        plt.close(fig)


rootPath = os.path.abspath('../../')
assetPath = rootPath + "/asset/"
# 使用示例
source_folder = rootPath + "/output/"
target_folder = assetPath + "/danger_trajectory/"
figure_folder = os.path.join(target_folder, 'figure')
columns_to_check = ['RearTTCRaw3', 'LeadTTCRaw3', 'LeftRearTTCRaw3', 'LeftLeadTTCRaw3', 'LeftAlongsideTTCRaw3']
value_range = (0, 5)

check_and_save_csv(source_folder, target_folder, figure_folder, columns_to_check, value_range)
