# -*- coding = utf-8 -*-
# @Time : 2024/11/25 20:05
# @Author : 王砚轩
# @File : DataExtraction.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
from loguru import logger
from src.figure.TTC_acc_figure import  create_output_folder

# 标识列
id_columns = ['recordingId', 'trackId', 'frame', 'MergingType']

# 输入压缩列
input_columns = ['xCenter', 'yCenter', 'heading', 'lonVelocity', 'latVelocity',
                 'lonAcceleration', 'latAcceleration', 'RearTTCRaw3', 'LeadTTCRaw3',
                 'LeftRearTTCRaw3', 'LeftLeadTTCRaw3', 'LeftAlongsideTTCRaw3']

ttc_columns = ['RearTTCRaw3', 'LeadTTCRaw3', 'LeftRearTTCRaw3', 'LeftLeadTTCRaw3', 'LeftAlongsideTTCRaw3']


def uniform_sampling(data, target_length=50):
    """
    等间隔抽取数据使其变为目标长度。

    :param data: 原始数据，类型为 numpy 数组或 Pandas DataFrame。
    :param target_length: 抽取后的目标长度。
    :return: 抽取后的数据，类型与输入一致。
    """
    indices = np.linspace(0, len(data) - 1, target_length, dtype=int)  # 生成等间隔索引
    return data.iloc[indices]


def main():
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    singleTrajPath = assetPath + "/single_traj/"
    outputPath = assetPath + "/extracted_data/"
    create_output_folder(assetPath, "extracted_data")
    # 获取文件夹中的所有 CSV 文件
    csv_files = [f for f in os.listdir(singleTrajPath) if f.endswith('.csv')]
    compress_data = []
    # 遍历文件并读取数据
    for file in csv_files:
        file_path = os.path.join(singleTrajPath, file)
        df = pd.read_csv(file_path)  # 读取 CSV 数据
        logger.info(f"{file} is Loading...")

        result = uniform_sampling(df)
        result.to_csv(outputPath+file)
        logger.info(f"Extracted single trajectory has been saved.")

        # 处理TTCRaw3列，当值小于0时设为999
        result.loc[:, ttc_columns] = result.loc[:, ttc_columns].applymap(lambda x: 999 if x < 0 else x)

        record = result[id_columns].iloc[0].tolist()

        for col in input_columns:
            record.extend(result[col].iloc[:50].tolist())

        compress_data.append(record)

    # 构建总体的DataFrame
    compressed_columns = id_columns
    for col in input_columns:
        compressed_columns.extend([f"{col}_{i + 1}" for i in range(50)])
    compressed_df = pd.DataFrame(compress_data, columns=compressed_columns)
    compressed_df.to_csv(assetPath+"compressed_data.csv")


if __name__ == '__main__':
    main()