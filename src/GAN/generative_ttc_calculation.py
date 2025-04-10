# -*- coding = utf-8 -*-
# @Time : 2025/4/9 16:23
# @Author : 王砚轩
# @File : generative_ttc_calculation.py
# @Software: PyCharm
import math
import os

import numpy as np
from config.laneletID import lanelet2data
from src.GAN.data_adjust_and_additional_calculation import get_group_by_recording_id
from loguru import logger
import pandas as pd
from src.figure.TTC_acc_figure import create_output_folder


def getTTC_V2(data):
    """
    计算两车 Time-To-Collision (TTC) 的一种近似：
      - data: pandas Series or dict with keys:
            'xCenter','yCenter','heading','xVelocity','yVelocity'
      - other_meta: pandas DataFrame (single-row) with columns ['length','width']
      - other_df: pandas DataFrame (single-row) with same columns as `data`
      - length, width: float, 当前车辆尺寸

    返回：
      - TTC（float），若无法计算则返回 -1。
    """
    def line_intersection_and_dist(P_list, Q_list):
        """遍历 P_list 的每个点，对 Q_list 边求交并计算距离。"""
        for pt in P_list:
            for i in range(len(Q_list)):
                p1, p2 = Q_list[i], Q_list[(i+1) % len(Q_list)]
                # 构造两条线的系数矩阵
                a1 = VA['y'] - VB['y']
                b1 = VB['x'] - VA['x']
                c1 = pt['x'] * (VA['y'] - VB['y']) + pt['y'] * (VB['x'] - VA['x'])
                a2 = p1['y'] - p2['y']
                b2 = p2['x'] - p1['x']
                c2 = p1['x'] * (p1['y'] - p2['y']) + p1['y'] * (p2['x'] - p1['x'])

                A = np.array([[a1, b1],
                              [a2, b2]])
                B = np.array([c1, c2])
                sol = np.linalg.solve(A, B)

                x_sol, y_sol = sol
                # 检查交点是否在边 p1-p2 上
                if (min(p1['x'], p2['x']) <= x_sol <= max(p1['x'], p2['x']) and
                    min(p1['y'], p2['y']) <= y_sol <= max(p1['y'], p2['y'])):
                    # 计算该交点到顶点 pt 的距离
                    d = math.hypot(pt['x'] - x_sol, pt['y'] - y_sol)
                    dist_list.append(d)

    lead_ttc, rear_ttc = 999, 999

    # 1) 计算当前车的 4 个顶点坐标
    location_id = get_group_by_recording_id(int(data['recordingId']))
    road_heading = float(lanelet2data.get(location_id, {}).get('heading', {}))
    theta_a = math.radians(data['ego_heading'] - road_heading)
    cx, cy = data['ego_lonLaneletPos'], data['ego_latLaneCenterOffset']
    length_a, width_a = data['ego_length'], data['ego_width']
    dx_l = 0.5 * length_a
    dy_w = 0.5 * width_a

    # 顶点 A1..A4
    alist = [
        {'x': cx + dx_l * math.cos(theta_a) + dy_w * math.sin(theta_a),
         'y': cy + dx_l * math.sin(theta_a) - dy_w * math.cos(theta_a)},
        {'x': cx - dx_l * math.cos(theta_a) + dy_w * math.sin(theta_a),
         'y': cy - dx_l * math.sin(theta_a) - dy_w * math.cos(theta_a)},
        {'x': cx - dx_l * math.cos(theta_a) - dy_w * math.sin(theta_a),
         'y': cy - dx_l * math.sin(theta_a) + dy_w * math.cos(theta_a)},
        {'x': cx + dx_l * math.cos(theta_a) - dy_w * math.sin(theta_a),
         'y': cy + dx_l * math.sin(theta_a) + dy_w * math.cos(theta_a)},
    ]

    # 2) 计算另一辆车的尺寸和顶点
    if data['lead_mask']:
        length_c = data['lead_length']
        width_c = data['lead_width']
        theta_c = math.radians(data['lead_heading'] - road_heading)
        ox = data['lead_lonLaneletPos']
        oy = data['lead_latLaneCenterOffset']
        odx_l = 0.5 * length_c
        ody_w = 0.5 * width_c

        clist = [
            {'x': ox + odx_l * math.cos(theta_c) + ody_w * math.sin(theta_c),
             'y': oy + odx_l * math.sin(theta_c) - ody_w * math.cos(theta_c)},
            {'x': ox - odx_l * math.cos(theta_c) + ody_w * math.sin(theta_c),
             'y': oy - odx_l * math.sin(theta_c) - ody_w * math.cos(theta_c)},
            {'x': ox - odx_l * math.cos(theta_c) - ody_w * math.sin(theta_c),
             'y': oy - odx_l * math.sin(theta_c) + ody_w * math.cos(theta_c)},
            {'x': ox + odx_l * math.cos(theta_c) - ody_w * math.sin(theta_c),
             'y': oy + odx_l * math.sin(theta_c) + ody_w * math.cos(theta_c)},
        ]

        # 3) 两车速度向量
        VA = {'x': data['ego_lonVelocity'] * math.cos(theta_a),
              'y': data['ego_lonVelocity'] * math.sin(theta_a)}
        VB = {'x': data['lead_lonVelocity'] * math.cos(theta_c),
              'y': data['lead_lonVelocity'] * math.sin(theta_c)}

        # 4) 寻找所有顶点对相交点到顶点的距离
        dist_list = []

        # A 顶点对 B 边，和 B 顶点对 A 边
        line_intersection_and_dist(alist, clist)
        line_intersection_and_dist(clist, alist)

        # 5) 计算并返回 TTC
        if dist_list:
            min_dist = min(dist_list)
            rel_speed = math.hypot(VA['x'] - VB['x'], VA['y'] - VB['y'])
            if rel_speed != 0:
                lead_ttc = min_dist / rel_speed

    if data['rear_mask']:
        length_c = data['rear_length']
        width_c = data['rear_width']
        theta_c = math.radians(data['rear_heading'] - road_heading)
        ox = data['rear_lonLaneletPos']
        oy = data['rear_latLaneCenterOffset']
        odx_l = 0.5 * length_c
        ody_w = 0.5 * width_c

        clist = [
            {'x': ox + odx_l * math.cos(theta_c) + ody_w * math.sin(theta_c),
             'y': oy + odx_l * math.sin(theta_c) - ody_w * math.cos(theta_c)},
            {'x': ox - odx_l * math.cos(theta_c) + ody_w * math.sin(theta_c),
             'y': oy - odx_l * math.sin(theta_c) - ody_w * math.cos(theta_c)},
            {'x': ox - odx_l * math.cos(theta_c) - ody_w * math.sin(theta_c),
             'y': oy - odx_l * math.sin(theta_c) + ody_w * math.cos(theta_c)},
            {'x': ox + odx_l * math.cos(theta_c) - ody_w * math.sin(theta_c),
             'y': oy + odx_l * math.sin(theta_c) + ody_w * math.cos(theta_c)},
        ]

        # 3) 两车速度向量
        VA = {'x': data['ego_lonVelocity'] * math.cos(theta_a),
              'y': data['ego_lonVelocity'] * math.sin(theta_a)}
        VB = {'x': data['rear_lonVelocity'] * math.cos(theta_c),
              'y': data['rear_lonVelocity'] * math.sin(theta_c)}

        # 4) 寻找所有顶点对相交点到顶点的距离
        dist_list = []

        # A 顶点对 B 边，和 B 顶点对 A 边
        line_intersection_and_dist(alist, clist)
        line_intersection_and_dist(clist, alist)

        # 5) 计算并返回 TTC
        if dist_list:
            min_dist = min(dist_list)
            rel_speed = math.hypot(VA['x'] - VB['x'], VA['y'] - VB['y'])
            if rel_speed != 0:
                rear_ttc = min_dist / rel_speed

    return lead_ttc, rear_ttc


def get_vehicle_dimensions(data, meta_path, poi_path):
    """
    从 meta CSV 和 poi CSV 中提取 ego/lead/rear 车辆的宽度和长度。

    参数：
      data: pandas Series，包含至少 ['recordingId','trackId'] 字段
      meta_path: str，meta CSV 文件路径，需包含 ['trackId','width','length']
      poi_path: str，poi CSV 文件路径，需包含 ['recordingId','trackId','leadId','rearId']

    返回：
      ego_width, ego_length,
      lead_width, lead_length,
      rear_width, rear_length
    若任一车辆信息未找到，则对应宽度/长度返回 None。
    """
    # 读取 CSV
    rec_id = int(data['recordingId'])
    trk_id = int(data['trackId'])

    file_path = os.path.join(meta_path, f"{rec_id}_tracksMeta.csv")
    meta_df = pd.read_csv(file_path)
    poi_df = pd.read_csv(poi_path)

    # 1) Ego
    ego_row = meta_df.loc[meta_df['trackId'] == trk_id]
    if len(ego_row) == 0:
        ego_width, ego_length = None, None
    else:
        ego_width = float(ego_row['width'].iloc[0])
        ego_length = float(ego_row['length'].iloc[0])

    # 2) 从 poi 表中找到该 rec_id & trk_id 的行
    poi_row = poi_df.loc[
        (poi_df['recordingId'] == rec_id) &
        (poi_df['trackId'] == trk_id)
        ]
    if len(poi_row) == 0:
        lead_id, rear_id = None, None
    else:
        lead_id = poi_row['leadId'].iloc[0]
        rear_id = poi_row['rearId'].iloc[0]

    # 3) Lead
    if pd.isna(lead_id):
        lead_width, lead_length = None, None
    else:
        lead_row = meta_df.loc[meta_df['trackId'] == lead_id]
        if len(lead_row) == 0:
            lead_width, lead_length = None, None
        else:
            lead_width = float(lead_row['width'].iloc[0])
            lead_length = float(lead_row['length'].iloc[0])

    # 4) Rear
    if pd.isna(rear_id):
        rear_width, rear_length = None, None
    else:
        rear_row = meta_df.loc[meta_df['trackId'] == rear_id]
        if len(rear_row) == 0:
            rear_width, rear_length = None, None
        else:
            rear_width = float(rear_row['width'].iloc[0])
            rear_length = float(rear_row['length'].iloc[0])

    return ego_width, ego_length, lead_width, lead_length, rear_width, rear_length


if __name__ == '__main__':
    rootPath = os.path.abspath('../../')
    assetPath = os.path.join(rootPath, 'asset')
    processPath = os.path.join(assetPath, 'processed_data')
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(processPath) if f.endswith('.csv')]
    create_output_folder(assetPath, 'generative_ttc')
    outputPath = os.path.join(assetPath, 'generative_ttc')
    meta_path = os.path.join(rootPath, 'drone-dataset-tools-master', 'data')
    poi_path = os.path.join(assetPath, 'mergingDataNew200m.csv')

    for filename in csv_files:
        file_path = os.path.join(processPath, filename)
        df = pd.read_csv(file_path)

        # 3. 用 lambda + apply，一次性把 6 个返回值展开到 6 列
        df[['ego_width', 'ego_length',
            'lead_width', 'lead_length',
            'rear_width', 'rear_length'
            ]] = df.apply(
            lambda row: get_vehicle_dimensions(row, meta_path, poi_path),
            axis=1,
            result_type='expand'
        )
        logger.info(f"{filename} size has been found.")
        df[['lead_ttc', 'rear_ttc']] = df.apply(
            lambda row: getTTC_V2(row),
            axis=1,
            result_type='expand'
        )
        logger.info(f"{filename} ttc has been calculated.")
        df.to_csv(os.path.join(outputPath, filename), index=False)