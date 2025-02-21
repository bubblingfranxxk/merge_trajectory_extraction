# -*- coding = utf-8 -*-
# @Time : 2025/2/7 16:23
# @Author : 王砚轩
# @File : multiVehicleExtraction.py
# @Software: PyCharm
import os
import pandas as pd
from loguru import logger
from pathlib import Path
from src.figure.TTC_acc_figure import create_output_folder
from src.GAN.data_adjust_and_additional_calculation import process_csv_files
import src.GAN.data_normalization as data_normalization


class MultiVehicleExtraction(object):
    def __init__(self, config):
        # 定义包含CSV文件的文件夹路径
        self.rootPath = os.path.abspath('../../')
        self.assetPath = self.rootPath + '/asset/'
        self.trajPath = self.assetPath + "/single_traj/"
        self.DISTANCE = config["distance_threshold"]
        self.poiPath = self.rootPath + "/asset/" + "mergingData" + str(self.DISTANCE) + "m.csv"
        self.trackPath = self.rootPath + '/output/'
        # 定义需要提取的列
        self.poi_selected_columns = [
            'recordingId',
            'trackId',
            'leadId',
            'rearId',
            'leftLeadId',
            'leftRearId',
            'leftAlongsideId'
        ]
        # 定于周围车辆标签
        self.surroundingVehilceLabels = ['rearId', 'leadId']
        # 单个轨迹
        self.traj = None
        self.maxFrame = None
        self.minFrame = None
        self.recordingId = None
        self.trackId = None
        self.surroundingPath = self.assetPath + '/surroundingTraj/'
        self.rear_folder = self.surroundingPath + '/rearId/'
        self.lead_folder = self.surroundingPath + '/leadId/'
        self.adjust_surroundingPath = self.assetPath + '/adjusted_surrounding/'
        self.adjust_rear_folder = self.adjust_surroundingPath + '/rearId/'
        self.adjust_lead_folder = self.adjust_surroundingPath + '/leadId/'
        self.normalize_surroundingPath = self.assetPath + '/normalization_surrounding/'
        self.normalize_rear_folder = self.normalize_surroundingPath + '/rearId/'
        self.normalize_lead_folder = self.normalize_surroundingPath + '/leadId/'
        self.feature_columns = ['lonLaneletPos', 'latLaneCenterOffset', 'heading', 'lonVelocity', 'lonAcceleration',
                                'latAcceleration']
        self.testPath = self.assetPath + "/extracted_data/"

    def run(self):
        # 直接读取所需的列
        poi_file = pd.read_csv(self.poiPath, usecols=self.poi_selected_columns)

        traj_folder = [f for f in Path(self.trajPath).glob('*.csv')]
        create_output_folder(self.assetPath, 'surroundingTraj')
        create_output_folder(self.surroundingPath, 'rearId')
        create_output_folder(self.surroundingPath, 'leadId')
        create_output_folder(self.assetPath, 'adjusted_surrounding')
        create_output_folder(self.adjust_surroundingPath, 'rearId')
        create_output_folder(self.adjust_surroundingPath, 'leadId')
        create_output_folder(self.assetPath, 'normalization_surrounding')
        create_output_folder(self.normalize_surroundingPath, 'rearId')
        create_output_folder(self.normalize_surroundingPath, 'leadId')


        # 读取并操作单个轨迹
        for file in traj_folder:
            self.traj = pd.read_csv(file)
            # 返回首尾帧号
            self.minFrame, self.maxFrame = self.traj['frame'].iloc[0], self.traj['frame'].iloc[-1]
            # 返回recording id 和 track id
            self.recordingId = self.traj['recordingId'].iloc[0]
            self.trackId = self.traj['trackId'].iloc[0]
            poi_traj = poi_file[(poi_file['recordingId'] == self.recordingId) & (poi_file['trackId'] == self.trackId)]
            if poi_traj.empty:
                logger.warning(f"self trajectory of recording id {self.recordingId} track id {self.trackId} not Found!")
                continue
            # 打开对应的othervehicle csv文件
            otherVehicleTraj = pd.read_csv(self.trackPath + f'/Output_{str(self.recordingId)}_{str(self.trackId)}_'
                                                            f'others_tracks.csv')

            for label in self.surroundingVehilceLabels:
                vehilceId = poi_traj[label].values[0]
                # logger.debug(f"{otherVehicleTraj['trackId'].dtype}")
                df = otherVehicleTraj[(otherVehicleTraj['trackId'] == vehilceId) &
                                      (otherVehicleTraj['frame'] >= self.minFrame) &
                                      (otherVehicleTraj['frame'] <= self.maxFrame)]
                df.to_csv(self.surroundingPath + f'/{label}/{self.recordingId}_{self.trackId}_{label}_trajectory.csv',
                          index=False)
                # logger.info(f'{self.recordingId}_{self.trackId}_{label} has done.')
        logger.info(f'surrroundingTraj is done.')

        process_csv_files(self.rear_folder, self.adjust_rear_folder)
        process_csv_files(self.lead_folder, self.adjust_lead_folder)
        self.read_csv_and_extract()
        data_normalization.main(self.adjust_rear_folder, data_normalization.recordingMapToLocation,
                                self.feature_columns, ttc_columns=None, path=self.normalize_surroundingPath,
                                folder='rearId')
        data_normalization.main(self.adjust_lead_folder, data_normalization.recordingMapToLocation,
                                self.feature_columns, ttc_columns=None, path=self.normalize_surroundingPath,
                                folder='leadId')

    def read_csv_and_extract(self):

        # 遍历目录中的所有文件
        for filename in os.listdir(self.adjust_rear_folder):
            if filename.endswith('.csv'):
                # 按'_'切分文件名
                parts = filename.split('_')
                # 确保文件名切割后前两部分是数字
                if not os.path.exists(f'{self.testPath}{parts[0]}_{parts[1]}_single_trajectory.csv'):
                    os.remove(self.adjust_rear_folder + filename)

        # 遍历目录中的所有文件
        for filename in os.listdir(self.adjust_lead_folder):
            if filename.endswith('.csv'):
                # 按'_'切分文件名
                parts = filename.split('_')
                # 确保文件名切割后前两部分是数字
                if not os.path.exists(f'{self.testPath}{parts[0]}_{parts[1]}_single_trajectory.csv'):
                    os.remove(self.adjust_lead_folder + filename)

