# -*- coding = utf-8 -*-
# @Time : 2024/12/12 20:24
# @Author : 王砚轩
# @File : data_adjust_and_additional_calculation.py
# @Software: PyCharm
import pandas as pd
from config.laneletID import lanelet2data  # 从 laneletID.py 中导入相关数据
from src.GAN.data_normalization import recordingMapToLocation
import os
from loguru import logger
from utils import common
import numpy as np
from src.figure.TTC_acc_figure import create_output_folder


def get_group_by_recording_id(recording_id):
    for group, ids in recordingMapToLocation.items():
        if recording_id in ids:
            return group
    return None


def process_csv_files(input_folder_path, output_folder_path):
    # 创建输出文件夹
    if output_folder_path and not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in os.listdir(input_folder_path):
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(input_folder_path, file_name)
            output_file_path = os.path.join(output_folder_path, file_name)

            # 读取 CSV 文件
            df = pd.read_csv(input_file_path)
            if df.empty:
                # logger.warning(f"dataframe {input_file_path} is None !")
                continue

            # 确保 lonLaneletPos 和 latLaneCenterOffset 列存在
            if 'lonLaneletPos' not in df or 'latLaneCenterOffset' not in df or 'recordingId' not in df \
                    or 'laneletId' not in df.columns or 'laneWidth' not in df.columns:
                logger.warning(f"File {file_name} is missing required columns. Skipping.")
                continue

            # 获取 recordingId 和对应的组
            recording_id = df['recordingId'].iloc[0]  # 假设 recordingId 对于一个文件是唯一的
            group = get_group_by_recording_id(recording_id)
            if not group:
                logger.warning(
                    f"RecordingId {recording_id} in file {file_name} does not belong to any group. Skipping.")
                continue

            # 操作 lonLaneletPos 列
            lane_length_offsets = lanelet2data.get(group, {}).get('lanelengthoffset', {})
            df['lonLaneletPos'] = df.apply(
                lambda row: common.processLaneletData(row['lonLaneletPos'], 'float'), axis=1
            )
            df['laneWidth'] = df.apply(
                lambda row: common.processLaneletData(row['laneWidth'], 'float'), axis=1
            )
            try:
                df['lonLaneletPos'] += df['laneletId'].map(lambda laneletId: lane_length_offsets[str(laneletId)])
            except Exception as e:
                logger.error(e)
                continue
            # 操作 latLaneCenterOffset 列
            lane_offsets = lanelet2data.get(group, {}).get('laneoffset', {})

            # 提取 lane_offsets 对应的值
            lane_offset = df['laneletId'].map(lambda x: lane_offsets[str(x)])

            # 根据 lane_offset 的正负生成符号（1 或 -1）
            sign = np.where(lane_offset >= 0, 1, -1)

            # 计算最终的偏移量
            df['latLaneCenterOffset'] += np.where(lane_offset != -1, lane_offset, 0) + sign * df['laneWidth'] / 2.0
            # df['latLaneCenterOffset'] += df['laneletId'].map(lambda laneletId: lane_offsets[str(laneletId)]) + \
            #                              df['laneWidth'] / 2.0

            # 保存处理后的文件
            df.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    folder_path = assetPath + "/single_traj/"  # 替换为你的文件夹路径
    outputPath = assetPath + "/adjusted_data/"
    create_output_folder(assetPath, 'adjusted_data')
    process_csv_files(folder_path, outputPath)
