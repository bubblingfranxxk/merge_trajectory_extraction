# -*- coding = utf-8 -*-
# @Time : 2023/12/26 20:10
# @Author : 王砚轩
# @File : mergingExtractionClass.py
# @Software: PyCharm

import os
import numpy as np
import pandas as pd
from loguru import logger
from config import laneletID
from utils import common
import shutil
import matplotlib.pyplot as plt


class MergingExtractionClass(object):
    def __init__(self, config):
        self.config = config
        self.TIMESTEP = config["timestep"]
        self.LOOKBACK = config["lookback"]
        self.DISTANCE = config["distance_threshold"]
        self.location = config["location_set"]

        self.rootPath = os.path.abspath('../../')
        self.savePath = os.path.abspath('../../') + "/asset/"
        self.outputPath = self.savePath + "mergingData" + str(self.DISTANCE) + "m.csv"

        self.usedColumns = ['recordingId', 'trackId', 'frame', 'trackLifetime', 'xCenter', 'yCenter', 'heading',
                            'width',
                            'length', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 'lonVelocity',
                            'latVelocity',
                            'lonAcceleration', 'latAcceleration', 'traveledDistance', 'latLaneCenterOffset',
                            'laneletId', 'laneChange', 'lonLaneletPos', 'leadId', 'rearId', 'leftLeadId', 'leftRearId',
                            'leftAlongsideId', 'curxglobalutm', 'curyglobalutm']

        self.HDMdata = None
        self.chosenLocation = None
        self.tracksSelf = None
        self.weekday = None
        self.otherVehicle = None
        self.locationTotalMergingDistance = {
            "2": 160.32,
            "3": 200.52,
            "5": 174.81,
            "6": 219.72
        }
        self.recordingMapToLocation = {
            "0": list(range(0, 19)),
            "1": list(range(19, 39)),
            "2": list(range(39, 53)),
            "3": list(range(53, 61)),
            "4": list(range(61, 73)),
            "5": list(range(73, 78)),
            "6": list(range(78, 93)),
        }

        self.surroundingVehiclesLabel = ["leadId",
                                         "rearId",
                                         "leftLeadId",
                                         "leftRearId",
                                         "leftAlongsideId",
                                         "rightLeadId",
                                         "rightRearId",
                                         "rightAlongsideId"]

    def readCsvFile(self, record):
        PathTracksMeta = self.rootPath + r"\drone-dataset-tools-master\data\{}_tracksMeta.csv".format(str(record))
        PathRecordingMeta = self.rootPath + r"\drone-dataset-tools-master\data\{}_recordingMeta.csv".format(
            str(record))
        PathTracks = self.rootPath + r"\drone-dataset-tools-master\data\{}_tracks.csv".format(str(record))

        tracks = pd.read_csv(PathTracks, low_memory=False)
        tracksMeta = pd.read_csv(PathTracksMeta)
        recordingMeta = pd.read_csv(PathRecordingMeta)

        logger.info("Loading recording {} ", record)
        # logger.info("Loading csv{}, {} and {}", PathTracksMeta, PathRecordingMeta, PathTracks)

        xUtmOrigin = recordingMeta["xUtmOrigin"].values[0]
        yUtmOrigin = recordingMeta["yUtmOrigin"].values[0]

        # 计算地理坐标系下的车辆中心的x,y 【为什么要算这个】
        tracks["curxglobalutm"] = tracks["xCenter"].apply(lambda x: x + xUtmOrigin)
        tracks["curyglobalutm"] = tracks["yCenter"].apply(lambda x: x + yUtmOrigin)

        tracks.fillna("-999", inplace=True)
        tracks.astype({"trackId": "int"}, )

        logger.info("Input {}_CSVFile successfully.", record)
        return tracks, tracksMeta, recordingMeta

    def create_output_folder(self, path, folder):
        output_path = os.path.join(path, folder)

        # 检查output文件夹是否已经存在
        if os.path.exists(output_path):
            # 如果已存在，则删除
            shutil.rmtree(output_path)

        # 创建output文件夹
        os.makedirs(output_path)

    def save_to_csv(self, dataframe, record1, record2=None):
        """
        将 DataFrame 保存为 CSV 文件

        参数：
        dataframe: 要保存的 DataFrame
        file_path: 文件路径，包括文件名和.csv扩展名
        """
        if record2 is None:
            outputPath = self.rootPath + r"\output\Output_{}_tracks.csv".format(str(record1))
            # shutil.rmtree(self.rootPath + r"\output")
            dataframe.to_csv(outputPath, index=False)  # 如果不想写入行索引，可以将index参数设置为False
            logger.info(f"DataFrame已保存至 {outputPath}")
        else:
            record = str(record1) + '_' + str(record2)
            outputPath = self.rootPath + r"\output\Output_{}_tracks.csv".format(record)
            # shutil.rmtree(self.rootPath + r"\output")
            print(outputPath)
            dataframe.to_csv(outputPath, index=False)  # 如果不想写入行索引，可以将index参数设置为False
            logger.info(f"DataFrame已保存至 {outputPath}")

    def getMergeTracks(self, tracks):
        arealist = self.HDMdata["mainlineUpstream"] + self.HDMdata["area123"] + self.HDMdata["area4"] + \
                   self.HDMdata["area5"]
        arealist = list(set(arealist))
        # print(f"arealist: {arealist}")
        temp = tracks[tracks["laneletId"].isin(arealist)]
        return temp

    '''
    def matchSurrodingVehicles(self, tracksMeta, row):
        if (False == row["MergingState"]) or (row["RouteClass"] == "mainline"):
            return "None", "None", "None", \
                   "None", "None", "None", \
                   "None", "None", "None", \
                   "None", "None", "None", \
                   "None", "None", "None", \
                   "None"
        trajectoryInfo = {}
        RearVehicleNumber = 0
        LeadVehicleNumber = 0
        MinimumRearDistance = 999
        MinimumLeadDistance = 999

        MinimumRearStatus = "None"
        MinimumLeadStatus = "None"
        MinimumRearClass = "None"
        MinimumLeadClass = "None"
        MergingType = "None"
        status = "None"

        RearVehicleSpeed = 999
        LeadVehicleSpeed = 999
        RearHeadway = 999
        LeadHeadway = 999

        LeadVehicleId = "None"
        RearVehicleId = "None"

        for vehicleType in self.surroundingVehiclesLabel:
            vehicleIdListUnique = self.tracksSelf[vehicleType].unique()
            for vehicleId in vehicleIdListUnique:

                currentInfo = self.otherVehicle[(self.otherVehicle["trackId"] == vehicleId) &
                                                (self.otherVehicle["frame"] == row["frame"])]
                if vehicleId == -1 or vehicleId == "-999" or currentInfo.empty:
                    continue

                distance = np.sqrt(np.square(row['xCenter'] - currentInfo['xCenter']) + np.square(
                    row['yCenter'] - currentInfo['yCenter']))
                speed = (row['xVelocity'] * currentInfo['xVelocity'])

                if distance > self.DISTANCE or speed < 0:
                    continue

                cursurrounidngrouteclass = self.checkVehicleRouteClass(
                    self.tracksBeforeMerge[self.tracksBeforeMerge["trackId"] == vehicleId])
                curLanelet2Id = [common.processLaneletData(x, "int") for x in curinfotail["laneletId"].unique()]

                if len(set(curLanelet2Id) & set(self.HDMdata["-1"])) != 0:
                    positionLabel = "on -1"
                elif len(set(curLanelet2Id) & set(self.HDMdata["-2"])) != 0:
                    positionLabel = "on -2"
                elif len(set(curLanelet2Id) & set(self.HDMdata["-3"])) != 0:
                    positionLabel = "on -3"
                elif len(set(curLanelet2Id) & set(self.HDMdata["entry"])) != 0:
                    positionLabel = "on entry"
                elif len(set(curLanelet2Id) & set(self.HDMdata["onramp"])) != 0:
                    positionLabel = "on onramp"
                else:
                    continue

                if (row["location"] == "2" or row["location"] == "3" or row[
                    "location"] == "5") and positionLabel != "on -2":
                    continue
                elif (row["location"] == "6") and positionLabel != "on -3":
                    continue

                # 有点乱，有点转懵了
                if vehicleId in self.alongsidetolead and vehicleType in ["leadId", "leftLeadId",
                                                                         "rightLeadId"] and vehicleId in self.tailLeadVehicles:
                    status = "alongside to lead"

                if vehicleId in self.alongsidetorear and vehicleType in ["rearId", "leftRearId",
                                                                         "rightRearId"] and vehicleId in self.tailRearVehicles:
                    status = "alongside to rear"

                if vehicleId in self.reartofront:
                    # 没整明白想干嘛？
                    leadvehicleFrame = np.append(np.argwhere(self.leadvehicles == vehicleId).flatten(),
                                                 np.argwhere(self.leftleadvehicles == vehicleId).flatten())
                    rearvehicleFrame = np.append(np.argwhere(self.rearvehicles == vehicleId).flatten(),
                                                 np.argwhere(self.leftRearvehicles == vehicleId).flatten())

                    if vehicleType in ["leadId", "leftLeadId", "rightLeadId"] and min(leadvehicleFrame) > max(
                            rearvehicleFrame) and vehicleId in self.tailLeadVehicles:
                        status = "rear to lead"
                    elif vehicleType in ["rearId", "leftRearId", "rightRearId"] and max(leadvehicleFrame) < min(
                            rearvehicleFrame) and vehicleId in self.tailRearVehicles:
                        status = "lead to rear"
                    else:
                        status = "WRONG"
                else:
                    status = "Exist"

                if vehicleType in ["rearId", "leftRearId", "rightRearId"] and vehicleId in self.tailRearVehicles:
                    if distance < MinimumRearDistance:
                        MinimumRearDistance = distance
                        MinimumRearStatus = status
                        MinimumRearClass = \
                            tracksmeta[tracksmeta["trackId"] == curinfotail["trackId"].values[0]]["class"].values[0]
                        RearVehicleId = vehicleId
                        RearVehicleSpeed = np.sqrt(
                            np.square(curinfotail['xVelocity'].mean()) + np.square(curinfotail['yVelocity'].mean()))
                        RearHeadway = MinimumRearDistance / RearVehicleSpeed

                elif vehicleType in ["leadId", "leftLeadId", "rightLeadId"] and vehicleId in self.tailLeadVehicles:
                    if distance < MinimumLeadDistance:
                        MinimumLeadDistance = distance
                        MinimumLeadStatus = status
                        MinimumLeadClass = \
                            tracksmeta[tracksmeta["trackId"] == curinfotail["trackId"].values[0]]["class"].values[0]
                        LeadVehicleId = vehicleId
                        LeadVehicleSpeed = np.sqrt(np.square(row['xVelocity']) + np.square(row['xVelocity']))
                        LeadHeadway = MinimumLeadDistance / LeadVehicleSpeed

                trajectoryInfo[str(vehicleType) + ":" + str(vehicleId)] = {
                    "id": curinfotail["trackId"].values.mean(),
                    "routeclass": cursurrounidngrouteclass,
                    "position": positionLabel,
                    "sidestatus": status,
                    "lonVelocity": curinfotail["lonVelocity"].values.mean(),
                    "latVelocity": curinfotail["latVelocity"].values.mean(),
                    "lonAcceleration": curinfotail["lonAcceleration"].values.mean(),
                    "latAcceleration": curinfotail["latAcceleration"].values.mean(),
                    "distance": distance,
                    "class":
                        tracksmeta[tracksmeta["trackId"] == curinfotail["trackId"].values[0]]["class"].values[
                            0],
                }

            surroudingInfo = {"vehicleNums": len(trajectoryInfo.keys()), "trajectory": trajectoryInfo}

            if MinimumRearStatus == "None" and MinimumLeadStatus == "None":
                MergingType = "A"
            elif MinimumRearStatus == "None" and MinimumLeadStatus == "Exist":
                MergingType = "B"
            elif MinimumRearStatus == "None" and MinimumLeadStatus == "rear to lead":
                MergingType = "C"
            elif MinimumRearStatus == "Exist" and MinimumLeadStatus == "None":
                MergingType = "D"
            elif MinimumRearStatus == "Exist" and MinimumLeadStatus == "Exist":
                MergingType = "E"
            elif MinimumRearStatus == "Exist" and MinimumLeadStatus == "rear to lead":
                MergingType = "F"
            elif MinimumRearStatus == "lead to rear" and MinimumLeadStatus == "None":
                MergingType = "G"
            elif MinimumRearStatus == "lead to rear" and MinimumLeadStatus == "Exist":
                MergingType = "H"

            return surroudingInfo, RearVehicleNumber, MinimumRearDistance, \
                   MinimumRearStatus, MinimumRearClass, LeadVehicleNumber, \
                   MinimumLeadDistance, MinimumLeadStatus, MinimumLeadClass, \
                   MergingType, LeadVehicleId, RearVehicleId, \
                   RearVehicleSpeed, RearHeadway, LeadVehicleSpeed, LeadHeadway
    '''

    def run(self):
        # self.create_output_folder(self.rootPath, 'output')
        self.create_output_folder(self.rootPath, 'asset')
        data = pd.DataFrame()
        locationTemp = [self.recordingMapToLocation[key] for key in self.location
                        if key in self.recordingMapToLocation]

        self.chosenLocation = [i for sublist in locationTemp for i in sublist]

        for record in range(0, 93):
            test_offset = []
            if record not in self.chosenLocation:
                continue

            tracks, tracksMeta, recordingMeta = self.readCsvFile("%02d" % record)
            location = str(recordingMeta["locationId"].values[0])
            weekday = recordingMeta["weekday"].values[0]
            self.HDMdata = laneletID.lanlet2data[location]
            self.location = location
            self.weekday = weekday

            # 把laneletid 为2个的数据取第一个
            tracks["latLaneCenterOffset"] = tracks.apply(
                lambda row: common.processLaneletData(row["latLaneCenterOffset"], "float"), axis=1)
            tracks["laneletId"] = tracks.apply(lambda row: common.processLaneletData(row["laneletId"], "int"), axis=1)
            tracks.set_index(keys="frame", inplace=True, drop=False)
            # self.save_to_csv(tracks, record)

            # 对tracks按照id分组，【curid是当前组内车辆id,curgroup当前组】
            for currentId, currentGroup in tracks[self.usedColumns].groupby("trackId"):

                currentGroup.sort_index(inplace=True)

                # 得到整条汇入轨迹
                self.tracksSelf = self.getMergeTracks(currentGroup)

                # 筛选掉没变道的轨迹，与出现时长小于3s的轨迹，以及没在汇入区的轨迹
                if (currentGroup['laneChange'].unique() == 0).all() or len(currentGroup) < 3 / self.TIMESTEP or \
                        not (np.any(np.isin(currentGroup["laneletId"].unique(), self.HDMdata["area1"]))):
                    continue

                maxIndex = max(self.tracksSelf["frame"])
                minIndex = min(self.tracksSelf["frame"])
                print(f"min: {minIndex}, max: {maxIndex}")

                test_offset.append(self.tracksSelf)

                # self.save_to_csv(self.tracksSelf, record, currentId)

                # 参考整条轨迹提取周围所有车轨迹
                self.otherVehicle = tracks[(tracks["frame"] >= minIndex) &
                                           (tracks["frame"] <= maxIndex)]
                # self.save_to_csv(self.otherVehicle, record, str(currentId) + "_others")

                # self.tracksSelf[
                #     ["SurroundingVehiclesInfo", "RearVehicleNumber", "MinimumRearDistance",
                #      "MinimumRearStatus", "MinimumRearClass", "LeadVehicleNumber",
                #      "MinimumLeadDistance", "MinimumLeadStatus", "MinimumLeadClass",
                #      "MergingType", "LeadVehicleId", "RearVehicleId",
                #      "MinimumRearSpeed", "MinimumRearHeadway", "MinimumLeadSpeed", "MinimumLeadHeadway"
                #      ]
                # ] = self.tracksSelf.apply(lambda row: self.match)

            common.plot_data_in_batches(test_offset, 10, record, self.savePath)
