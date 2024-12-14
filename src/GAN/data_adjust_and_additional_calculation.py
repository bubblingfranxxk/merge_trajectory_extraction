# -*- coding = utf-8 -*-
# @Time : 2024/12/12 20:24
# @Author : 王砚轩
# @File : data_adjust_and_additional_calculation.py
# @Software: PyCharm
import pandas as pd
from config.laneletID import lanlet2data  # 从 laneletID.py 中导入相关数据
from src.GAN.data_normalization import recordingMapToLocation
import os
from loguru import logger


def get_group_by_recording_id(recording_id):
    for group, ids in recordingMapToLocation.items():
        if recording_id in ids:
            return group
    return None


def process_csv_files(input_folder_path, output_folder_path):
    # 创建输出文件夹
    if output_folder_path and not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(input_folder_path, file_name)
            output_file_path = os.path.join(output_folder_path, file_name)

            # 读取 CSV 文件
            df = pd.read_csv(input_file_path)

            # 确保 traveledDistance 和 latLaneCenterOffset 列存在
            if 'traveledDistance' not in df or 'latLaneCenterOffset' not in df or 'recordingId' not in df or 'laneletId' not in df.columns:
                logger.warning(f"File {file_name} is missing required columns. Skipping.")
                continue

            # 操作 traveledDistance 列
            df['traveledDistance'] -= df['traveledDistance'].iloc[0]

            # 获取 recordingId 和对应的组
            recording_id = df['recordingId'].iloc[0]  # 假设 recordingId 对于一个文件是唯一的
            group = get_group_by_recording_id(recording_id)

            if not group:
                logger.warning(f"RecordingId {recording_id} in file {file_name} does not belong to any group. Skipping.")
                continue

            # 操作 latLaneCenterOffset 列
            lane_offsets = lanlet2data.get(group, {}).get('laneoffset', {})
            df['latLaneCenterOffset'] += df['laneletId'].map(lambda laneletId: lane_offsets.get(str(laneletId), 0))

            # 保存处理后的文件
            df.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    folder_path = assetPath + "/single_traj/"  # 替换为你的文件夹路径
    outputPath = assetPath + "/adjusted_data/"
    process_csv_files(folder_path, outputPath)
