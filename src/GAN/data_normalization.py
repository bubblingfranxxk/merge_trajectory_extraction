# -*- coding = utf-8 -*-
# @Time : 2024/12/11 21:23
# @Author : 王砚轩
# @File : data_normalization.py
# @Software: PyCharm
import os
import pandas as pd
from loguru import logger
import numpy as np
recordingMapToLocation = {
    "2": list(range(39, 53)),
    "3": list(range(53, 61)),
    "5": list(range(73, 78)),
    "6": list(range(78, 93))
}


def get_group(recording_id, recording_map):
    """
    根据 recordingId 判断文件属于哪个分组。
    """
    for group, ids in recording_map.items():
        if recording_id in ids:
            return group
    return None


def compute_global_stats(dataframes, feature_columns):
    """
    计算指定列的全局均值和标准差。
    """
    combined_data = pd.concat([df[feature_columns] for df in dataframes if set(feature_columns).issubset(df.columns)],
                              axis=0)
    global_mean = combined_data.mean()
    global_std = combined_data.std()
    return global_mean, global_std


def custom_normalize_ttc(ttc_values):
    """
    自定义归一化公式：exp(-x)。
    """

    # 将负值转换为999
    ttc_values = np.where(ttc_values < 0, 999, ttc_values)
    return np.exp(-ttc_values)


def process_group_data(folder_path, group_files, feature_columns, ttc_columns, output_folder=None):
    """
    对某组文件执行标准化和归一化操作。
    """
    # 加载数据
    dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in group_files]

    # 计算全局统计值
    global_mean, global_std = compute_global_stats(dataframes, feature_columns)

    # 处理每个文件
    for file, df in zip(group_files, dataframes):
        # 标准化 feature_columns
        if set(feature_columns).issubset(df.columns):
            df[feature_columns] = (df[feature_columns] - global_mean) / global_std

        # 归一化 ttc_columns
        if set(ttc_columns).issubset(df.columns):
            df[ttc_columns] = df[ttc_columns].apply(custom_normalize_ttc)

        # 保存文件
        output_path = os.path.join(output_folder or folder_path, file)
        df.to_csv(output_path, index=False)
        logger.debug(f"已处理文件: {output_path}")


def main(folder_path, recording_map, feature_columns, ttc_columns, output_folder=None):
    """
    主函数：按 recordingId 对 CSV 文件分组，并对每组数据执行标准化和归一化。
    """
    # 创建输出文件夹
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件列表并按分组分类
    group_files = {group: [] for group in recording_map.keys()}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            recording_id = int(df['recordingId'].iloc[0])  # 假设 recordingId 在每个文件中固定

            group = get_group(recording_id, recording_map)
            if group:
                group_files[group].append(file_name)

    # 对每个分组执行处理
    for group, files in group_files.items():
        logger.info(f"处理分组: {group}, 文件数量: {len(files)}")
        process_group_data(folder_path, files, feature_columns, ttc_columns, output_folder)


if __name__ == "__main__":
    # 主函数调用示例
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    folder_path = assetPath + "/extracted_data/"
    output_folder = assetPath + "/normalized_data/"  # 如果覆盖原文件，将此参数设为None
    feature_columns = ['traveledDistance', 'latLaneCenterOffset', 'heading', 'lonVelocity', 'lonAcceleration', 'latAcceleration']
    ttc_columns = ['RearTTCRaw3', 'LeadTTCRaw3', 'LeftRearTTCRaw3', 'LeftLeadTTCRaw3', 'LeftAlongsideTTCRaw3']

    # 第二步：对数据进行标准化和归一化
    main(folder_path, recordingMapToLocation, feature_columns, ttc_columns, output_folder)
