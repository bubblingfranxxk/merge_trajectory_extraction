# -*- coding = utf-8 -*-
# @Time : 2025/3/31 22:01
# @Author : 王砚轩
# @File : post_process.py
# @Software: PyCharm
import os
import json
import numpy as np
import pandas as pd
from src.GAN.data_normalization import recordingMapToLocation
from loguru import logger

def kalman_filter(data, process_variance=1e-5, measurement_variance=0.1):
    """
    对一维时间序列数据进行卡尔曼滤波去噪和平滑处理
    :param data: 1D numpy 数组
    :param process_variance: 过程噪声方差
    :param measurement_variance: 观测噪声方差
    :return: 平滑后的数组
    """
    n = len(data)
    xhat = np.zeros(n)  # 状态估计值
    P = np.zeros(n)  # 估计误差协方差
    xhat[0] = data[0]
    P[0] = 1.0
    Q = process_variance
    R = measurement_variance
    for k in range(1, n):
        # 预测
        xhat_minus = xhat[k - 1]
        P_minus = P[k - 1] + Q

        # 更新
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (data[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return xhat


def get_location_id(recording_id):
    """
    根据 recordingMapToLocation 字典判断 recording_id 应归属的 locationId。
    recordingMapToLocation 定义如下：
      "0": 0 ~ 18,
      "1": 19 ~ 38,
      "2": 39 ~ 52,
      "3": 53 ~ 60,
      "4": 61 ~ 72,
      "5": 73 ~ 77,
      "6": 78 ~ 92
    :param recording_id: 录音/轨迹的编号，能够转换为整数
    :return: 对应的 locationId 字符串；若没有匹配则返回 None
    """
    try:
        rec_int = int(recording_id)
        for loc_id, rec_range in recordingMapToLocation.items():
            if rec_int in rec_range:
                return loc_id
        return None
    except Exception as e:
        print(f"转换 recordingId {recording_id} 时出错: {e}")
        return None


def get_stats_for_column(col, location_id, ego_stats, lead_stats, rear_stats):
    """
    根据列名前缀判断使用哪个 JSON 统计信息字典，
    然后根据 location_id 获取对应的均值和标准差数据。
    :param col: 特征列名称，如 "ego_lonLaneletPos"、"lead_heading"、"rear_lonVelocity" 等
    :param location_id: 当前分组对应的 location_id（字符串）
    :param ego_stats: ego_statics.json 读取后的字典
    :param lead_stats: lead_statics.json 读取后的字典
    :param rear_stats: rear_statics.json 读取后的字典
    :return: 对应统计信息字典，或 None
    """
    if col.startswith("ego_"):
        return ego_stats.get(location_id, None)
    elif col.startswith("lead_"):
        return lead_stats.get(location_id, None)
    elif col.startswith("rear_"):
        return rear_stats.get(location_id, None)
    else:
        return None


def process_group(df, location_id, ego_stats, lead_stats, rear_stats):
    """
    对单个分组（矩阵）进行后处理：
      1. 对除 recordingId 和 trackId 外的各个特征列先进行卡尔曼滤波去噪平滑，
      2. 根据对应 JSON 文件中（依据列名前缀和 location_id 获取）的均值和标准差进行反标准化处理
         （若对应值为 null，则跳过反标准化）。
    :param df: DataFrame，包含 recordingId, trackId 及其它特征列
    :param location_id: 当前组的 location_id
    :param ego_stats: ego 统计信息字典
    :param lead_stats: lead 统计信息字典
    :param rear_stats: rear 统计信息字典
    :return: 处理后的 DataFrame
    """
    processed = df.copy()
    feature_columns = [col for col in df.columns if col not in ["recordingId", "trackId"]]
    # logger.debug(feature_columns)
    for col in feature_columns:
        smoothed = kalman_filter(df[col].values)
        stats_for_col = get_stats_for_column(col, location_id, ego_stats, lead_stats, rear_stats)
        # logger.debug(stats_for_col)
        base_col = col.split("_", 1)[-1]
        if stats_for_col is not None and \
           stats_for_col.get("mean", {}).get(base_col) is not None and \
           stats_for_col.get("std", {}).get(base_col) is not None:
            mean = stats_for_col["mean"][base_col]
            std = stats_for_col["std"][base_col]
            denorm = smoothed * std + mean
        else:
            denorm = smoothed
        processed[col] = denorm
    return processed


def modify_based_on_merging_type(df, merging_type):
    """
    根据 MergingType 对数据进行特殊赋值处理：
      - 对于 MergingType 为 A 的数据，将包含 lead 字段和 rear 字段的列赋值为 999；
      - 对于 MergingType 为 B 或 C 的数据，将包含 rear 字段的列赋值为 999；
      - 对于 MergingType 为 D 的数据，将包含 lead 字段的列赋值为 999。
    :param df: DataFrame，包含各字段
    :param merging_type: 字符串，例如 "A", "B", "C", "D"
    :return: 修改后的 DataFrame
    """
    lead_cols = [col for col in df.columns if col.startswith("lead_")]
    rear_cols = [col for col in df.columns if col.startswith("rear_")]
    if merging_type == "A":
        df[lead_cols] = 999
        df[rear_cols] = 999
    elif merging_type in ["B", "C"]:
        df[rear_cols] = 999
    elif merging_type == "D":
        df[lead_cols] = 999
    return df


def main():
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + '/asset/'
    g_data = assetPath + '/GENERATED_DATA/'
    # 设置文件路径，可根据需要修改
    generated_csv = g_data + "fake_data_epoch_199.csv"  # 生成数据的 CSV 文件
    original_data_folder = assetPath + "/normalized_data/"  # 存储原始单轨迹数据的文件夹，文件名格式如 "39_65_single_trajectory.csv"
    ego_stats_json = original_data_folder + "statistic_data.json"  # 存储均值和标准差的 JSON 文件
    lead_stats_json = assetPath + "/normalization_surrounding/leadId/" + "statistic_data.json"
    rear_stats_json = assetPath + "/normalization_surrounding/rearId/" + "statistic_data.json"
    output_folder = assetPath + "processed_data"
    os.makedirs(output_folder, exist_ok=True)

    # 读取三个 JSON 文件
    with open(ego_stats_json, "r") as f:
        ego_stats = json.load(f)
    with open(lead_stats_json, "r") as f:
        lead_stats = json.load(f)
    with open(rear_stats_json, "r") as f:
        rear_stats = json.load(f)
    # 读取生成数据 CSV 文件
    df = pd.read_csv(generated_csv)
    # 检查必须包含 recordingId 和 trackId 列
    if "recordingId" not in df.columns or "trackId" not in df.columns:
        raise ValueError("CSV 文件中缺少 recordingId 和 trackId 列。")

    # 按 recordingId 和 trackId 复合索引拆分数据
    grouped = df.groupby(["recordingId", "trackId"])

    # 对每个分组进行后处理，并分别保存
    for (recording_id, track_id), group in grouped:
        # group 中可能无明显时间序列顺序，可根据需要进行排序（例如按照行索引或其它时间列）
        group = group.sort_index()
        # 根据 recordingId 获取 locationId
        location_id = get_location_id(recording_id)
        if location_id is None:
            print(f"recordingId {recording_id} 无法映射到 locationId，跳过该组。")
            continue

        # 对该分组数据进行卡尔曼滤波与反标准化处理
        processed_group = process_group(group, location_id, ego_stats, lead_stats, rear_stats)

        # 根据 recordingId 和 trackId 构造原始数据文件名，例如 "39_65_single_trajectory.csv"
        orig_filename = f"{recording_id}_{track_id}_single_trajectory.csv"
        orig_path = os.path.join(original_data_folder, orig_filename)
        if not os.path.exists(orig_path):
            print(f"原始数据文件 {orig_path} 不存在，跳过该组。")
            continue

        # 读取原始数据，获取原始数据长度 l 和 MergingType（假定该字段存在且各行相同）
        orig_df = pd.read_csv(orig_path)
        l = len(orig_df)
        if "MergingType" not in orig_df.columns:
            print(f"原始数据文件 {orig_path} 缺少 MergingType 列，跳过该组。")
            continue
        merging_type = orig_df["MergingType"].iloc[0]

        # 删除超出原始数据长度 l 的数据
        processed_group = processed_group.iloc[:l].copy()

        # 根据 MergingType 对部分字段赋值 999
        processed_group = modify_based_on_merging_type(processed_group, merging_type)

        # 构造输出文件名，例如 "39_65_track_XXX_processed.csv"
        output_filename = f"{recording_id}_track_{track_id}_processed.csv"
        output_path = os.path.join(output_folder, output_filename)
        processed_group.to_csv(output_path, index=False)
        print(f"保存处理后数据: recordingId={recording_id}, trackId={track_id}, MergingType={merging_type} 到 {output_path}")


if __name__ == "__main__":
    main()
