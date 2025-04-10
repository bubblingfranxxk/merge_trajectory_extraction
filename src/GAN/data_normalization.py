# -*- coding = utf-8 -*-
# @Time : 2024/12/11 21:23
# @Author : 王砚轩
# @File : data_normalization.py
# @Software: PyCharm
import os
import pandas as pd
from loguru import logger
import numpy as np
from src.figure.TTC_acc_figure import create_output_folder
import json

recordingMapToLocation = {
    "0": list(range(19)),
    "1": list(range(19, 39)),
    "2": list(range(39, 53)),
    "3": list(range(53, 61)),
    "4": list(range(61, 73)),
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
    # logger.debug(dataframes)
    combined_data = pd.concat([df[feature_columns] for df in dataframes if set(feature_columns).issubset(df.columns)],
                              axis=0)
    global_mean = combined_data.mean()
    global_std = combined_data.std()
    return global_mean, global_std


def custom_normalize_ttc(ttc_values):
    """
    自定义归一化公式：1-2/pi*arctan(x)。
    """

    # 将负值转换为999
    ttc_values = np.where(ttc_values < 0, 999, ttc_values)
    return 1 - 2 / np.pi * np.arctan(ttc_values)


def process_group_data(folder_path, group_files, feature_columns, ttc_columns=None, output_folder=None):
    """
    对某组文件执行标准化和归一化操作。
    """
    # logger.debug(folder_path)
    # logger.debug(group_files)
    # 加载数据
    dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in group_files]
    # logger.debug(f"a: {dataframes}")

    # 计算全局统计值
    global_mean, global_std = compute_global_stats(dataframes, feature_columns)

    # 处理每个文件
    for file, df in zip(group_files, dataframes):
        # 标准化 feature_columns
        if set(feature_columns).issubset(df.columns):
            df[feature_columns] = (df[feature_columns] - global_mean) / global_std

        if ttc_columns:
            # 归一化 ttc_columns
            if set(ttc_columns).issubset(df.columns):
                df[ttc_columns] = df[ttc_columns].apply(custom_normalize_ttc)

        # 保存文件
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, index=False)
    logger.info(f"已处理文件: {output_folder}")

    return global_mean, global_std


def main(folder_path, recording_map, feature_columns, ttc_columns=None, path=None, folder=None):
    """
    主函数：按 recordingId 对 CSV 文件分组，并对每组数据执行标准化和归一化。
    """
    # 创建输出文件夹
    create_output_folder(path, folder)
    # logger.debug(folder_path)
    # logger.debug(os.listdir(folder_path))
    max_length = 0
    # 获取文件列表并按分组分类
    group_files = {group: [] for group in recording_map.keys()}
    group_data = {group:
                      {'mean': None,
                       'std': None}
                  for group in recording_map.keys()}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # logger.debug(file_path)
            df = pd.read_csv(file_path)
            max_length = max(len(df), max_length)
            recording_id = int(df['recordingId'].iloc[0])  # 假设 recordingId 在每个文件中固定

            group = get_group(recording_id, recording_map)
            if group:
                group_files[group].append(file_name)
    # logger.debug(f"b: {group_files}")
    # logger.debug(f"c: {group_data}")
    # logger.debug(group_files.items())
    # 对每个分组执行处理
    for group, files in group_files.items():
        if len(files) == 0:
            continue
        logger.info(f"处理分组: {group}, 文件数量: {len(files)}")
        group_data[group]['mean'], group_data[group]['std'] = \
            process_group_data(folder_path, files, feature_columns, ttc_columns, path + f'/{folder}/')
    # logger.debug(group_data)
    # 将 group_data 中的均值和标准差转换为字典，便于保存为 JSON
    group_data_serializable = {}
    for group, stats in group_data.items():
        group_data_serializable[group] = {
            'mean': stats['mean'].to_dict() if stats['mean'] is not None else None,
            'std': stats['std'].to_dict() if stats['std'] is not None else None
        }

    # 保存为 JSON 文件，文件名为 statistic_data.json
    json_name = 'statistic_data.json'
    json_path = os.path.join(path, folder, json_name)
    # logger.debug(json_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(group_data_serializable, f, indent=4)
    logger.info(f"已保存全局统计数据到: {json_path}")
    logger.info(f"Max length is {max_length}.")


if __name__ == "__main__":
    # 主函数调用示例
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    folder_path = assetPath + "/extracted_data/"
    output_name = "normalized_data"
    output_folder = assetPath + "/normalized_data/"  # 如果覆盖原文件，将此参数设为None
    feature_columns = ['lonLaneletPos', 'latLaneCenterOffset', 'heading', 'lonVelocity', 'lonAcceleration',
                       'latAcceleration']
    ttc_columns = ['RearTTCRaw3', 'LeadTTCRaw3', 'LeftRearTTCRaw3', 'LeftLeadTTCRaw3', 'LeftAlongsideTTCRaw3']

    # 第二步：对数据进行标准化和归一化
    main(folder_path, recordingMapToLocation, feature_columns, ttc_columns, assetPath, output_name)
