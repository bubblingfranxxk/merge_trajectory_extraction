# -*- coding = utf-8 -*-
# @Time : 2024/7/1 9:50
# @Author : 王砚轩
# @File : advanced_TTC_calculate_and_figure.py
# @Software: PyCharm
import os
import pandas as pd
from loguru import logger
from utils import common
from TTC_acc_figure import dtype_spec
import math
CO_A = 1.5
CO_B = 1.3


def save_to_csv(dataframe, outputPath, record1, record2=None):
    """
    将 DataFrame 保存为 CSV 文件

    参数：
    dataframe: 要保存的 DataFrame
    file_path: 文件路径，包括文件名和.csv扩展名
    """
    if record2 is None:
        outputPath = outputPath + r"\Output_{}_tracks.csv".format(str(record1))
        # shutil.rmtree(self.rootPath + r"\output")
        dataframe.to_csv(outputPath, index=False)  # 如果不想写入行索引，可以将index参数设置为False
        logger.info(f"DataFrame已保存至 {outputPath}")
    else:
        record = str(record1) + '_' + str(record2)
        outputPath = outputPath + r"\Output_{}_tracks.csv".format(record)
        # shutil.rmtree(self.rootPath + r"\output")
        # print(outputPath)
        dataframe.to_csv(outputPath, index=False)  # 如果不想写入行索引，可以将index参数设置为False
        logger.info(f"DataFrame已保存至 {outputPath}")


def main():
    root_path = os.path.abspath('../../')
    asset_path = root_path + "/asset/"
    # 读取exception文件
    exception_path = asset_path + "exception.csv"
    exception_data = pd.read_csv(exception_path)
    # 输出路径
    output_path = root_path + "/output/"

    # 周围车辆类型
    surrounding_vehicles_label = ["leadId",
                                  "rearId",
                                  "leftLeadId",
                                  "leftRearId",
                                  "leftAlongsideId"]

    def get_TTC_V3(this_data, other_vehicle_this_frame, guess_value):
        heading = math.radians(this_data['heading'])
        other_heading = math.radians(other_vehicle_this_frame['heading'])
        a1 = CO_A * this_data['length'] * 0.5
        b1 = CO_B * this_data['width'] * 0.5
        x1 = this_data['xCenter']
        y1 = this_data['yCenter']
        vx = other_vehicle_this_frame['xVelocity'].values[0] - this_data['xVelocity']
        vy = other_vehicle_this_frame['yVelocity'].values[0] - this_data['yVelocity']
        a2 = CO_A * other_vehicle_this_frame['length'].values[0] * 0.5
        b2 = CO_B * other_vehicle_this_frame['width'].values[0] * 0.5
        x2 = other_vehicle_this_frame['xCenter'].values[0]
        y2 = other_vehicle_this_frame['yCenter'].values[0]

        result = common.ellipses_tangent_time(a1, b1, heading, x1, y1, vx, vy, a2, b2, other_heading, x2, y2,
                                              guess_value)
        t_solution, xt, yt = result
        if t_solution < 5:
            result = common.reload_ellipses_tangent_time(a1, b1, heading, x1, y1, vx, vy, a2, b2, other_heading, x2, y2)
            t_solution, xt, yt = result
        return t_solution

    for index, row in exception_data.iterrows():
        traj_string = "Output_" + str(row['recordingId']) + "_" + str(row['trackId'])\
                    + "_tracks.csv"
        other_traj_string = "Output_" + str(row['recordingId']) + "_" + str(row['trackId'])\
                            + "_others_tracks.csv"
        traj_path = root_path + "/output/" + traj_string
        other_traj_path = root_path + "/output/" + other_traj_string
        traj = pd.read_csv(traj_path, dtype=dtype_spec)
        other_traj = pd.read_csv(other_traj_path, dtype=dtype_spec)
        logger.info(f"{traj_string} has been imported.")

        for frame, data in traj.iterrows():
            for vehicle_type in surrounding_vehicles_label:
                other_vehicle_this_frame = other_traj[(other_traj['trackId'] == data[vehicle_type]) &
                                                      (other_traj['frame'] == data['frame'])]
                if vehicle_type == "rearId" and data['RearVehicleId'] > 0 and data['RearDeltaV'] < 0:
                    data['RearTTCRaw3'] = get_TTC_V3(data, other_vehicle_this_frame,
                                                     -data['RearDistance'] / data['RearDeltaV'])
                if vehicle_type == "leadId" and data['LeadVehicleId'] > 0 and data['LeadDeltaV'] > 0:
                    data['LeadTTCRaw3'] = get_TTC_V3(data, other_vehicle_this_frame,
                                                     data['LeadDistance'] / data['LeadDeltaV'])
                if vehicle_type == "leftRearId" and data['LeftRearVehicleId'] > 0 and data['LeftRearDeltaV'] < 0:
                    data['LeftRearTTCRaw3'] = get_TTC_V3(data, other_vehicle_this_frame,
                                                         -data['LeftRearDistance'] / data['LeftRearDeltaV'])
                if vehicle_type == "leftLeadId" and data['LeftLeadVehicleId'] > 0 and data['LeftLeadDeltaV'] > 0:
                    try:
                        temp = get_TTC_V3(data, other_vehicle_this_frame,
                                          data['LeftLeadDistance'] / data['LeftLeadDeltaV'])
                        data['LeftLeadTTCRaw3'] = temp
                    except:
                        logger.warning("printing !")
                        logger.warning(data)
                        logger.warning(other_vehicle_this_frame)
                        logger.warning(data['LeftLeadDistance'] / data['LeftLeadDeltaV'])
                if vehicle_type == "leftAlongsideId" and data['LeftAlongsideVehicleId'] > 0:
                    data['LeftAlongsideTTCRaw3'] = \
                        get_TTC_V3(data, other_vehicle_this_frame,
                                   abs(data['LeftAlongsideDistance'] / data['LeftAlongsideDeltaV']))
        save_to_csv(traj, output_path, row['recordingId'], row['trackId'])




if __name__ == '__main__':
    main()