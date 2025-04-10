# -*- coding = utf-8 -*-
# @Time : 2025/3/23 23:38
# @Author : 王砚轩
# @File : plot_merge_traj.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import pandas as pd
import os
from loguru import logger


def plot_lat_lane_center(input_path, file_list, save_path):
    """
    根据file_list列表读取csv文件，将每个文件的latLaneCenterOffset列绘制成子图
    按每行4个子图排列，最终保存到指定路径

    参数:
        file_list (list): CSV文件路径列表
        save_path (str): 结果图片保存路径
    """

    num_files = len(file_list)
    cols = 4
    rows = (num_files + cols - 1) // cols  # 计算需要的行数

    # 创建子图画布
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)  # 确保单行时保持二维结构
    axes_flat = axes.flatten()

    for i, file_path in enumerate(file_list):
        ax = axes_flat[i]
        file_num = file_path.split('_')
        logger.debug(file_num)
        try:
            # 读取CSV文件
            df = pd.read_csv(input_path+file_path)
            # 提取数据并绘制
            letter = chr(97 + i)  # 97是ASCII码的'a'
            lat_data = df['latLaneCenterOffset']
            ax.plot(lat_data, color='blue', linewidth=3)

            # 设置子图标题和标签
            ax.set_title(f"recording ID = {file_num[0]}, \ntrack ID = {file_num[1]}", fontsize=24)
            ax.set_xlabel(f'({letter}) Frame', fontsize=24)
            ax.set_ylabel('latLaneCenterOffset', fontsize=24)

            # 调整刻度字体大小
            ax.tick_params(axis='both', which='major', labelsize=18)
        except Exception as e:
            logger.info(f"Error processing {file_path}: {str(e)}")
            ax.set_title(f"Error: {os.path.basename(file_path)}", color='red')
            continue

    # 隐藏多余的空子图
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
    logger.info(f"图表已保存至：{save_path}")


# 示例用法
if __name__ == '__main__':
    # 配置参数
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"  # 合并后的输出文件
    singleTrajPath = assetPath + '/single_traj/'
    outputPath = assetPath + 'traj_latlaneoffset.png'
    file_config = {
        'recordingId': [39, 52, 53, 60, 73, 77, 78, 92],
        'track': [960, 5, 28, 25, 15, 3, 28, 1055]
    }

    # 使用zip组合两个列表，并用列表推导式生成结果
    result = [
        f"{recording_id}_{track_id}_single_trajectory.csv"
        for recording_id, track_id
        in zip(file_config['recordingId'], file_config['track'])
    ]

    plot_lat_lane_center(singleTrajPath, result, outputPath)