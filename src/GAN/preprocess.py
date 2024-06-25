# -*- coding = utf-8 -*-
# @Time : 2024/6/2 22:09
# @Author : 王砚轩
# @File : preprocess.py
# @Software: PyCharm
import os
from src.figure.TTC_acc_figure import create_output_folder
from src.figure.TTC_acc_figure import dtype_spec
from loguru import logger
import pandas as pd
import numpy as np


def main(level):
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    if level == 1:
        # 获取文件夹中合并的MergingTrajectory
        traj_files = [f for f in os.listdir(assetPath) if f.endswith('.csv')]
        # 构建输出路径
        create_output_folder(assetPath, "Preprocess_data1")
        outputpath = assetPath + r"/Preprocess_data1/"

        for file in traj_files:
            if "Trajectory" not in file:
                continue
            file_path = assetPath + file
            df = pd.read_csv(file_path, dtype=dtype_spec)
            logger.info("Loading {}", file)

            # 初始化存储DataFrame的列表
            dataframes = []
            start_i = 0

            # 遍历DataFrame的每一行，从第二行开始
            for i in range(1, len(df)):
                # 当前行与上一行的recordingId和trackId
                current_recordingId = df.iloc[i]['recordingId']
                current_trackId = df.iloc[i]['trackId']
                previous_recordingId = df.iloc[i - 1]['recordingId']
                previous_trackId = df.iloc[i - 1]['trackId']

                # 检查分割条件
                if not (current_trackId == previous_trackId and current_recordingId == previous_recordingId):
                    segment = df.iloc[start_i:i:5].reset_index(drop=True)
                    dataframes.append(segment)
                    segment.to_csv(outputpath + f"{previous_recordingId}_{previous_trackId}_data.csv", index=False)
                    logger.info(f"recording id {previous_recordingId}, "
                                f"track id {previous_trackId} segment is finished.")
                    start_i = i

                previous_recordingId = current_recordingId
                previous_trackId = current_trackId

            # 最后一段数据添加到列表中
            segment = df.iloc[start_i::5].reset_index(drop=True)
            segment.to_csv(outputpath + f"{previous_recordingId}_{previous_trackId}_data.csv", index=False)
            dataframes.append(segment)

            # 打印分割后的DataFrames数量
            logger.info(f'Number of segments: {len(dataframes)}')

    elif level == 2:
        filepath = assetPath + r"/PreProcess_data1/"
        outputpath = assetPath + r"/PreProcess_data2/"
        create_output_folder(assetPath, "Preprocess_data2")
        # 定义时间窗口大小
        window_size = 15
        # 获取文件夹中的MergingTrajectory
        traj_files = [f for f in os.listdir(filepath) if f.endswith('.csv')]
        # 总数据
        total_compress_data = []

        for file in traj_files:
            # 创建一个新的 DataFrame 来存储压缩后的数据
            compressed_data = []
            file_spilt = file.split('_')
            recordingId = int(file_spilt[0])
            trackId = int(file_spilt[1])
            df = pd.read_csv(filepath + file, dtype=dtype_spec)
            logger.info(f"Loading recordingId {recordingId}, trackId {trackId} ...")
            # logger.debug(df)
            # 标识列
            id_columns = ['recordingId', 'trackId', 'frame']

            # 输入压缩列
            input_columns = ['xCenter', 'yCenter', 'heading', 'lonVelocity', 'latVelocity',
                             'lonAcceleration', 'latAcceleration', 'RearTTCRaw3', 'LeadTTCRaw3',
                             'LeftRearTTCRaw3', 'LeftLeadTTCRaw3', 'LeftAlongsideTTCRaw3']

            # 输出压缩列
            output_columns = ['xCenter', 'yCenter', 'heading']

            # 处理TTCRaw3列，当值小于0时设为999
            ttc_columns = ['RearTTCRaw3', 'LeadTTCRaw3', 'LeftRearTTCRaw3', 'LeftLeadTTCRaw3', 'LeftAlongsideTTCRaw3']
            df[ttc_columns] = df[ttc_columns].applymap(lambda x: 999 if x < 0 else x)

            # 遍历 DataFrame，每次读取3s
            for i in range(len(df) - window_size + 1):
                window = df.iloc[i:i + window_size]

                # 取窗口的第一行的标识列数据
                record = window[id_columns].iloc[0].tolist()

                # 输入压缩列数据（前10行）
                for col in input_columns:
                    record.extend(window[col].iloc[:10].tolist())

                # 输出压缩列数据（后5行）
                for col in output_columns:
                    record.extend(window[col].iloc[-5:].tolist())

                compressed_data.append(record)

            # 构建总体的DataFrame
            compressed_columns = id_columns
            for col in input_columns:
                compressed_columns.extend([f"{col}_{i + 1}" for i in range(10)])
            for col in output_columns:
                compressed_columns.extend([f"{col}_out_{i + 1}" for i in range(5)])

            compressed_df = pd.DataFrame(compressed_data, columns=compressed_columns)

            filename = f"{recordingId}_{trackId}_compressed_data.csv"
            # 保存每条轨迹CSV文件
            compressed_df.to_csv(outputpath + filename, index=False)
            total_compress_data.extend(compressed_data)

        # 保存总体CSV文件
        total_compress_data_df = pd.DataFrame(total_compress_data, columns=compressed_columns)
        total_compress_data_df.to_csv(assetPath+"compressed_data.csv")


if __name__ == '__main__':
    # main(1)
    main(2)
