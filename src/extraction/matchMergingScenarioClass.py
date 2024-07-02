# -*- coding = utf-8 -*-
# @Time : 2024/5/27 16:55
# @Author : 王砚轩
# @File : matchMergingScenarioClass.py
# @Software: PyCharm

import os
import pandas as pd
from loguru import logger


class matchScenariosClass(object):
    def __init__(self, config):
        # 定义包含CSV文件的文件夹路径
        self.rootPath = os.path.abspath('../../')
        self.trajPath = self.rootPath + "/output/"
        self.DISTANCE = config["distance_threshold"]
        self.poiPath = self.rootPath + "/asset/" + "mergingData" + str(self.DISTANCE) + "m.csv"
        self.outputPath = self.rootPath + "/asset/"
        # 日志文件
        self.logPath = self.rootPath + "/log/logfile.txt"
        logger.add(self.logPath, rotation="500 MB")

    def run(self):
        # 获取文件夹中所有CSV文件的文件名
        csv_files = [f for f in os.listdir(self.trajPath) if f.endswith('.csv')]
        # 获取汇入场景CSV
        scenarios = pd.read_csv(self.poiPath)

        # 初始化一个空的DataFrame列表
        dataframes = []
        mergingPoint_new = []

        # 读取每个CSV文件并添加到DataFrame列表中
        for file in csv_files:
            matchStat = True
            file_path = os.path.join(self.trajPath, file)
            pathWords = file_path.split('_')
            if "others" in pathWords:
                continue
            recordingId = int(pathWords[3])
            trackId = int(pathWords[4])
            logger.info("{} is loading ...", file_path)
            logger.info("recording id is {}, track id is {}", recordingId, trackId)

            # 匹配汇入轨迹与汇入场景
            trajectory = pd.read_csv(file_path)
            thisScenario = pd.DataFrame()
            for index, row in scenarios.iterrows():
                if row['recordingId'] == recordingId and row['trackId'] == trackId:
                    thisScenario = pd.DataFrame(row).transpose()
                    mergingPoint_new.append(thisScenario)
                    matchStat = False
                    break
            if matchStat:
                logger.warning("this trajectory isn't matched. Recording id is {}, track id is {}.",
                               recordingId, trackId)
                continue
            # print(thisScenario)
            trajectory['MergingType'] = thisScenario['MergingType'].iloc[0]
            dataframes.append(trajectory)

        # 如果需要，可以将所有DataFrame合并为一个
        all_traj = pd.concat(dataframes, ignore_index=True)
        all_scen = pd.concat(mergingPoint_new, ignore_index=True)

        # 保存
        trajPath = self.outputPath + "mergingTrajectory" + str(self.DISTANCE) + "m.csv"
        scenPath = self.outputPath + "mergingDataNew" + str(self.DISTANCE) + "m.csv"
        all_traj.to_csv(trajPath, index=False)  # 如果不想写入行索引，可以将index参数设置为False
        logger.info(f"DataFrame已保存至 {trajPath}")
        all_scen.to_csv(scenPath, index=False)  # 如果不想写入行索引，可以将index参数设置为False
        logger.info(f"DataFrame已保存至 {scenPath}")
