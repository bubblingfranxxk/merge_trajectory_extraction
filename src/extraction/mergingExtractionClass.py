# -*- coding = utf-8 -*-
# @Time : 2023/12/26 20:10
# @Author : 王砚轩
# @File : mergingExtractionClass.py
# @Software: PyCharm
import math
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
        self.savemode = config["savemode"]

        self.rootPath = os.path.abspath('../../')
        self.savePath = os.path.abspath('../../') + "/result/"
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
        self.laneChangeFrame = None
        self.tracksSelf = None
        self.weekday = None
        self.otherVehicle = None
        self.Meta = None
        self.length = None
        self.width = None
        self.vehicleType = None

        self.leadVehicle = None  # 整个汇入过程中的前车
        self.leadDeltaX = None
        self.leadDeltaV = None
        self.leadDeltaAcce = None
        self.leadTTC_V1 = None
        self.leadTTC_V2 = None
        self.leadTTC_V3 = None
        self.rearVehicle = None  # 整个汇入过程中的后车
        self.rearDeltaX = None
        self.rearDeltaV = None
        self.rearDeltaAcce = None
        self.rearTTC_V1 = None
        self.rearTTC_V2 = None
        self.rearTTC_V3 = None
        self.leftleadVehicle = None  # 整个汇入过程中的侧前车
        self.leftleadDeltaX = None
        self.leftleadDeltaV = None
        self.leftleadDeltaAcce = None
        self.leftleadTTC_V1 = None
        self.leftleadTTC_V2 = None
        self.leftleadTTC_V3 = None
        self.leftrearVehicle = None  # 整个汇入过程中的侧后车
        self.leftrearDeltaX = None
        self.leftrearDeltaV = None
        self.leftrearDeltaAcce = None
        self.leftrearTTC_V1 = None
        self.leftrearTTC_V2 = None
        self.leftrearTTC_V3 = None
        self.leftalongsideVehicle = None  # 整个汇入过程中的侧车
        self.leftalongsideDeltaX = None
        self.leftalongsideDeltaV = None
        self.leftalongsideDeltaAcce = None
        self.leftalongsideTTC_V1 = None
        self.leftalongsideTTC_V2 = None
        self.leftalongsideTTC_V3 = None

        # TTC-V3的系数
        self.co_a = 1.5
        self.co_b = 1.2

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
                                         "leftAlongsideId"]

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

    def save_to_csv(self, dataframe, outputPath, record1, record2=None):
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

    def getMergeTracks(self, tracks):
        arealist = self.HDMdata["mainlineUpstream"] + self.HDMdata["area123"] + self.HDMdata["area4"] + \
                   self.HDMdata["area5"]
        arealist = list(set(arealist))
        # print(f"arealist: {arealist}")
        temp = tracks[tracks["laneletId"].isin(arealist)]
        return temp

    def matchSurroundingVehicles(self, row, tracksMeta):
        trajectoryInfo = {}

        self.rearVehicle = 0
        self.rearDeltaX = 999
        self.rearDeltaV = 999
        self.rearDeltaAcce = 999
        self.rearTTC_V1 = 999
        self.rearTTC_V2 = 999
        self.rearTTC_V3 = 999
        self.leadVehicle = 0
        self.leadDeltaX = 999
        self.leadDeltaV = 999
        self.leadDeltaAcce = 999
        self.leadTTC_V1 = 999
        self.leadTTC_V2 = 999
        self.leadTTC_V3 = 999
        self.leftleadVehicle = 0
        self.leftleadDeltaX = 999
        self.leftleadDeltaV = 999
        self.leftleadDeltaAcce = 999
        self.leftleadTTC_V1 = 999
        self.leftleadTTC_V2 = 999
        self.leftleadTTC_V3 = 999
        self.leftrearVehicle = 0
        self.leftrearDeltaX = 999
        self.leftrearDeltaV = 999
        self.leftrearDeltaAcce = 999
        self.leftrearTTC_V1 = 999
        self.leftrearTTC_V2 = 999
        self.leftrearTTC_V3 = 999
        self.leftalongsideVehicle = 0
        self.leftalongsideDeltaX = 999
        self.leftalongsideDeltaV = 999
        self.leftalongsideDeltaAcce = 999
        self.leftalongsideTTC_V1 = 999
        self.leftalongsideTTC_V2 = 999
        self.leftalongsideTTC_V3 = 999

        self.Meta = None
        self.length = 999
        self.width = 999
        RearHeadway = 999
        LeadHeadway = 999
        status = "None"
        # 汇入车辆的属性
        self.Meta = tracksMeta[tracksMeta['trackId'] == row['trackId']]
        self.length = self.Meta['length'].values[0]
        self.width = self.Meta['width'].values[0]
        self.vehicleType = self.Meta['class'].values[0]

        for vehicleType in self.surroundingVehiclesLabel:
            # print(vehicleType)
            # 周围车辆的属性
            otherVehicleMeta = tracksMeta[tracksMeta['trackId'] == row[vehicleType]]
            # otherLength = otherVehicleMeta['length'].values[0]
            # otherWidth = otherVehicleMeta['width'].values[0]
            # otherType = otherVehicleMeta['class'].values[0]

            otherVehicleThisFrame = self.otherVehicle[(self.otherVehicle['trackId'] == row[vehicleType]) &
                                                      (self.otherVehicle['frame'] == row['frame'])]
            if otherVehicleThisFrame.empty or row[vehicleType] == -1 or row[vehicleType] == "-999":
                continue
            # print('xcenter:')
            # print(row['xCenter'])
            distance = np.sqrt(np.square(row['xCenter'] - otherVehicleThisFrame['xCenter'].values[0]) + np.square(
                row['yCenter'] - otherVehicleThisFrame['yCenter'].values[0]))
            speed = (row['xVelocity'] * otherVehicleThisFrame['xVelocity'].values[0])
            if distance > self.DISTANCE or speed < 0:
                continue

            curLanelet2Id = [common.processLaneletData(x, "int") for x in otherVehicleThisFrame["laneletId"].unique()]
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
                logger.warning("Can't recognize other vehicle position label.")
                continue

            if (row["recordingId"] in self.recordingMapToLocation["2"] or row["recordingId"] in self.
                    recordingMapToLocation["3"] or row["recordingId"] in self.recordingMapToLocation["5"]) \
                    and positionLabel != "on -2":
                continue
            elif row["recordingId"] in self.recordingMapToLocation["6"] and positionLabel != "on -3":
                continue

            status = "Exist"
            theta = math.radians(self.HDMdata["heading"])
            # 速度差和加速度差统一为汇入车-其他车
            deltaV = -(otherVehicleThisFrame["xVelocity"].values[0] * math.cos(theta) + otherVehicleThisFrame[
                "yVelocity"].values[0] * math.sin(theta)) + row["xVelocity"] * math.cos(theta) + row["yVelocity"] * \
                math.sin(theta)
            deltaAcce = -(otherVehicleThisFrame["xAcceleration"].values[0] * math.cos(theta) + otherVehicleThisFrame[
                    "yAcceleration"].values[0] * math.sin(theta)) + row["xAcceleration"] * math.cos(theta) + row[
                                         "yAcceleration"] * math.sin(theta)
            thisFrame = self.laneChangeFrame["frame"].values[0]
            if vehicleType == "leadId":
                self.leadVehicle = row[vehicleType]
                self.leadDeltaX = distance
                self.leadDeltaV = deltaV
                self.leadDeltaAcce = deltaAcce
                if self.leadVehicle and self.leadDeltaV > 0:
                    self.leadTTC_V2 = self.getTTC_V2(row, otherVehicleMeta, otherVehicleThisFrame)
                    self.leadTTC_V3 = self.getTTC_V3(row, otherVehicleMeta, otherVehicleThisFrame)
            elif vehicleType == "rearId":
                self.rearVehicle = row[vehicleType]
                self.rearDeltaX = distance
                self.rearDeltaV = deltaV
                self.rearDeltaAcce = deltaAcce
                if self.rearVehicle and self.rearDeltaV < 0:
                    self.rearTTC_V2 = self.getTTC_V2(row, otherVehicleMeta, otherVehicleThisFrame)
                    self.rearTTC_V3 = self.getTTC_V3(row, otherVehicleMeta, otherVehicleThisFrame)
            elif row["frame"] < thisFrame and vehicleType == "leftLeadId":
                self.leftleadVehicle = row[vehicleType]
                self.leftleadDeltaX = distance
                self.leftleadDeltaV = deltaV
                self.leftleadDeltaAcce = deltaAcce
                if self.leftleadVehicle and self.leftleadDeltaV > 0:
                    self.leftleadTTC_V2 = self.getTTC_V2(row, otherVehicleMeta, otherVehicleThisFrame)
                    self.leftleadTTC_V3 = self.getTTC_V3(row, otherVehicleMeta, otherVehicleThisFrame)
            elif row["frame"] < thisFrame and vehicleType == "leftRearId":
                self.leftrearVehicle = row[vehicleType]
                self.leftrearDeltaX = distance
                self.leftrearDeltaV = deltaV
                self.leftrearDeltaAcce = deltaAcce
                if self.leftrearVehicle and self.leftrearDeltaV < 0:
                    self.leftrearTTC_V2 = self.getTTC_V2(row, otherVehicleMeta, otherVehicleThisFrame)
                    self.leftrearTTC_V3 = self.getTTC_V3(row, otherVehicleMeta, otherVehicleThisFrame)
            elif row["frame"] < thisFrame and vehicleType == "leftAlongsideId":
                self.leftalongsideVehicle = row[vehicleType]
                self.leftalongsideDeltaX = distance
                self.leftalongsideDeltaV = deltaV
                self.leftalongsideDeltaAcce = deltaAcce
                if self.leftalongsideVehicle:
                    self.leftalongsideTTC_V2 = self.getTTC_V2(row, otherVehicleMeta, otherVehicleThisFrame)
                    self.leftalongsideTTC_V3 = self.getTTC_V3(row, otherVehicleMeta, otherVehicleThisFrame)

            trajectoryInfo[str(vehicleType) + ":" + str(row[vehicleType])] = {
                # "id": otherVehicleThisFrame["trackId"],
                "position": positionLabel,
                # "lonVelocity": otherVehicleThisFrame["lonVelocity"].values[0],
                # "latVelocity": otherVehicleThisFrame["latVelocity"].values[0],
                # "lonAcceleration": otherVehicleThisFrame["lonAcceleration"].values[0],
                # "latAcceleration": otherVehicleThisFrame["latAcceleration"].values[0],
                # "distance": distance,
                "class":
                    tracksMeta[tracksMeta["trackId"] == otherVehicleThisFrame["trackId"].values[0]]["class"].values[0],
            }
        self.getTTC_V1()

        surroundingInfo = {"vehicleNums": len(trajectoryInfo.keys()), "trajectory": trajectoryInfo}
        return surroundingInfo, self.rearVehicle, self.rearDeltaX, self.rearDeltaV, self.rearDeltaAcce, \
            self.rearTTC_V1, self.rearTTC_V2, self.rearTTC_V3, \
            self.leadVehicle, self.leadDeltaX, self.leadDeltaV, self.leadDeltaAcce, self.leadTTC_V1, self.leadTTC_V2, \
            self.leadTTC_V3, \
            self.leftrearVehicle, self.leftrearDeltaX, self.leftrearDeltaV, self.leftrearDeltaAcce, \
            self.leftrearTTC_V1, self.leftrearTTC_V2, self.leftrearTTC_V3, \
            self.leftleadVehicle, self.leftleadDeltaX, self.leftleadDeltaV, self.leftleadDeltaAcce, \
            self.leftleadTTC_V1, self.leftleadTTC_V2, self.leftrearTTC_V3, \
            self.leftalongsideVehicle, self.leftalongsideDeltaX, self.leftalongsideDeltaV, \
            self.leftalongsideDeltaAcce, self.leftalongsideTTC_V1, self.leftalongsideTTC_V2, self.leftalongsideTTC_V3

    def getTTC_V1(self):
        if self.leadVehicle and self.leadDeltaV > 0:
            self.leadTTC_V1 = self.leadDeltaX / self.leadDeltaV
        if self.rearVehicle and self.rearDeltaV < 0:
            self.rearTTC_V1 = - self.rearDeltaX / self.rearDeltaV
        if self.leftleadVehicle and self.leftleadDeltaV > 0:
            self.leftleadTTC_V1 = self.leftleadDeltaX / self.leftleadDeltaV
        if self.leftrearVehicle and self.leftrearDeltaV < 0:
            self.leftrearTTC_V1 = - self.leftrearDeltaX / self.leftrearDeltaV
        if self.leftalongsideVehicle:
            self.leftalongsideTTC_V1 = self.leftalongsideDeltaX / abs(self.leftalongsideDeltaV)

    def getTTC_V2(self, data, otherMeta, otherVehicleThisFrame):
        heading = math.radians(data['heading'])
        alist = []

        AX1 = data['xCenter'] + 0.5 * self.length * math.cos(heading) + 0.5 * self.width * math.sin(heading)
        AY1 = data['yCenter'] + 0.5 * self.length * math.sin(heading) - 0.5 * self.width * math.cos(heading)
        A1 = {'x': AX1, 'y': AY1}
        alist.append(A1)

        AX2 = data['xCenter'] - 0.5 * self.length * math.cos(heading) + 0.5 * self.width * math.sin(heading)
        AY2 = data['yCenter'] - 0.5 * self.length * math.sin(heading) - 0.5 * self.width * math.cos(heading)
        A2 = {'x': AX2, 'y': AY2}
        alist.append(A2)

        AX3 = data['xCenter'] - 0.5 * self.length * math.cos(heading) - 0.5 * self.width * math.sin(heading)
        AY3 = data['yCenter'] - 0.5 * self.length * math.sin(heading) + 0.5 * self.width * math.cos(heading)
        A3 = {'x': AX3, 'y': AY3}
        alist.append(A3)

        AX4 = data['xCenter'] + 0.5 * self.length * math.cos(heading) - 0.5 * self.width * math.sin(heading)
        AY4 = data['yCenter'] + 0.5 * self.length * math.sin(heading) + 0.5 * self.width * math.cos(heading)
        A4 = {'x': AX4, 'y': AY4}
        alist.append(A4)

        other_heading = math.radians(otherVehicleThisFrame['heading'])
        length_other = otherMeta['length'].values[0]
        width_other = otherMeta['width'].values[0]
        blist = []

        BX1 = otherVehicleThisFrame['xCenter'].values[0] + 0.5 * length_other * math.cos(other_heading) \
            + 0.5 * width_other * math.sin(other_heading)
        BY1 = otherVehicleThisFrame['yCenter'].values[0] + 0.5 * length_other * math.sin(other_heading) \
            - 0.5 * width_other * math.cos(other_heading)
        B1 = {'x': BX1, 'y': BY1}
        blist.append(B1)

        BX2 = otherVehicleThisFrame['xCenter'].values[0] - 0.5 * length_other * math.cos(other_heading) \
            + 0.5 * width_other * math.sin(other_heading)
        BY2 = otherVehicleThisFrame['yCenter'].values[0] - 0.5 * length_other * math.sin(other_heading) \
            - 0.5 * width_other * math.cos(other_heading)
        B2 = {'x': BX2, 'y': BY2}
        blist.append(B2)

        BX3 = otherVehicleThisFrame['xCenter'].values[0] - 0.5 * length_other * math.cos(other_heading) \
            - 0.5 * width_other * math.sin(other_heading)
        BY3 = otherVehicleThisFrame['yCenter'].values[0] - 0.5 * length_other * math.sin(other_heading) \
            + 0.5 * width_other * math.cos(other_heading)
        B3 = {'x': BX3, 'y': BY3}
        blist.append(B3)

        BX4 = otherVehicleThisFrame['xCenter'].values[0] + 0.5 * length_other * math.cos(other_heading) \
            - 0.5 * width_other * math.sin(other_heading)
        BY4 = otherVehicleThisFrame['yCenter'].values[0] + 0.5 * length_other * math.cos(other_heading) \
            + 0.5 * width_other * math.cos(other_heading)
        B4 = {'x': BX4, 'y': BY4}
        blist.append(B4)

        VAX = data['xVelocity']
        VAY = data['yVelocity']
        VA = {'x': VAX, 'y': VAY}
        VBX = otherVehicleThisFrame['xVelocity'].values[0]
        VBY = otherVehicleThisFrame['yVelocity'].values[0]
        VB = {'x': VBX, 'y': VBY}

        blist.append(B1)
        alist.append(A1)

        dist_list = []
        for point in alist:
            for i in range(1, len(blist)):
                a1 = VA['y'] - VB['y']
                b1 = VB['x'] - VA['x']
                c1 = point['x'] * (VA['y'] - VB['y']) + point['y'] * (VB['x'] - VA['x'])
                a2 = blist[i-1]['y'] - blist[i]['y']
                b2 = blist[i]['x'] - blist[i-1]['x']
                c2 = blist[i-1]['x'] * (blist[i-1]['y'] - blist[i]['y']) + blist[i-1]['y'] * (blist[i]['x'] -
                                                                                              blist[i-1]['x'])
                A = np.array([[a1, b1],
                             [a2, b2]])
                B = np.array([c1, c2])
                solution = np.linalg.solve(A, B)
                min_x = min(b['x'] for b in blist[:-1])
                max_x = max(b['x'] for b in blist[:-1])
                min_y = min(b['y'] for b in blist[:-1])
                max_y = max(b['y'] for b in blist[:-1])
                # print(min_x, max_x, min_y, max_y)
                if max_x >= solution[0] >= min_x and max_y >= solution[1] >= min_y:
                    dist = math.sqrt((point['x'] - solution[0])**2 + (point['y'] - solution[1])**2)
                    dist_list.append(dist)

        for point in blist:
            for i in range(1, len(alist)):
                a1 = VA['y'] - VB['y']
                b1 = VB['x'] - VA['x']
                c1 = point['x'] * (VA['y'] - VB['y']) + point['y'] * (VB['x'] - VA['x'])
                a2 = alist[i-1]['y'] - alist[i]['y']
                b2 = alist[i]['x'] - alist[i-1]['x']
                c2 = alist[i-1]['x'] * (alist[i-1]['y'] - alist[i]['y']) + alist[i-1]['y'] * (alist[i]['x'] -
                                                                                              alist[i-1]['x'])
                A = np.array([[a1, b1],
                              [a2, b2]])
                B = np.array([c1, c2])
                solution = np.linalg.solve(A, B)
                min_x = min(b['x'] for b in alist[:-1])
                max_x = max(b['x'] for b in alist[:-1])
                min_y = min(b['y'] for b in alist[:-1])
                max_y = max(b['y'] for b in alist[:-1])
                if max_x >= solution[0] >= min_x and max_y >= solution[1] >= min_y:
                    dist = math.sqrt((point['x'] - solution[0]) ** 2 + (point['y'] - solution[1]) ** 2)
                    dist_list.append(dist)
        if dist_list:
            return min(dist_list)/math.sqrt((VA['x'] - VB['x'])**2 + (VA['y'] - VB['y'])**2)
        else:
            return -1

    def getTTC_V3(self, data, otherMeta, otherVehicleThisFrame):
        heading = math.radians(data['heading'])
        other_heading = math.radians(otherVehicleThisFrame['heading'])
        a1 = self.co_a * data['length'] * 0.5
        b1 = self.co_b * data['width'] * 0.5
        x1 = data['xCenter']
        y1 = data['yCenter']
        vx = otherVehicleThisFrame['xVelocity'].values[0] - data['xVelocity']
        vy = otherVehicleThisFrame['yVelocity'].values[0] - data['yVelocity']
        a2 = self.co_a * otherVehicleThisFrame['length'].values[0] * 0.5
        b2 = self.co_b * otherVehicleThisFrame['width'].values[0] * 0.5
        x2 = otherVehicleThisFrame['xCenter'].values[0]
        y2 = otherVehicleThisFrame['yCenter'].values[0]

        result = common.ellipses_tangent_time(a1, b1, heading, x1, y1, vx, vy, a2, b2, other_heading, x2, y2)
        if result is not None:
            t_solution, xt, yt = result
            return t_solution

    def run(self):
        if self.savemode == 'test':
            self.create_output_folder(self.rootPath, 'output')
            path = self.rootPath+"/output/"
        else:
            self.create_output_folder(self.rootPath, 'result')
            path = self.savePath
        # self.create_output_folder(self.rootPath, 'asset')

        locationTemp = [self.recordingMapToLocation[key] for key in self.location
                        if key in self.recordingMapToLocation]

        self.chosenLocation = [i for sublist in locationTemp for i in sublist]

        for record in range(0, 93):
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

            # 对tracks按照id分组，【curid是当前组内车辆id,curgroup当前组】
            for currentId, currentGroup in tracks[self.usedColumns].groupby("trackId"):

                currentGroup.sort_index(inplace=True)

                # 得到整条汇入轨迹
                self.tracksSelf = self.getMergeTracks(currentGroup)
                condition = (self.tracksSelf['laneChange'] == 1) & \
                            (self.tracksSelf['latLaneCenterOffset'] > self.tracksSelf['latLaneCenterOffset'].shift(1))

                # 筛选掉没变道的轨迹，与出现时长小于3s的轨迹，没在汇入区的轨迹， 以及变道到道路边界【即向右变道】的轨迹
                if (self.tracksSelf["laneChange"].unique() == 0).all() or len(currentGroup) < 3 / self.TIMESTEP or \
                        not np.any(np.isin(currentGroup["laneletId"].unique(), self.HDMdata["area1"])) or \
                        any(condition):
                    continue
                self.laneChangeFrame = self.tracksSelf[self.tracksSelf['laneChange'] == 1]
                # print("lanechange Frame:")
                # print(self.laneChangeFrame["frame"].values[0])
                maxIndex = max(self.tracksSelf["frame"])
                minIndex = min(self.tracksSelf["frame"])
                # print(f"min: {minIndex}, max: {maxIndex}")

                # self.save_to_csv(self.tracksSelf, record, currentId)

                # 参考整条轨迹提取周围所有车轨迹
                self.otherVehicle = tracks[(tracks["frame"] >= minIndex) &
                                           (tracks["frame"] <= maxIndex)]
                self.save_to_csv(self.otherVehicle, path, record, str(currentId) + "_others")

                tracksSelf_c = self.tracksSelf.copy()
                tracksSelf_c[
                    ["SurroundingVehiclesInfo",
                     "RearVehicleId", "RearDistance", "RearDeltaV", "RearDeltaAcceleration", "RearTTCRaw1",
                     "RearTTCRaw2", "RearTTCRaw3",
                     "LeadVehicleId", "LeadDistance", "LeadDeltaV", "LeadDeltaAcceleration", "LeadTTCRaw1",
                     "LeadTTCRaw2", "LeadTTCRaw3",
                     "LeftRearVehicleId", "LeftRearDistance", "LeftRearDeltaV", "LeftRearDeltaAcceleration",
                     "LeftRearTTCRaw1", "LeadRearTTCRaw2", "LeftRearTTCRaw3",
                     "LeftLeadVehicleId", "LeftLeadDistance", "LeftLeadDeltaV", "LeftLeadDeltaAcceleration",
                     "LeftLeadTTCRaw1", "LeftLeadTTCRaw2", "LeftLeadTTCRaw3",
                     "LeftAlongsideVehicleId", "LeftAlongsideDistance", "LeftAlongsideDeltaV",
                     "LeftAlongsideDeltaAcceleration", "LeftAlongsideTTCRaw1", "LeftAlongsideTTCRaw2",
                     "LeftAlongsideTTCRaw3"
                     ]
                ] = tracksSelf_c.apply(lambda row: self.matchSurroundingVehicles(row, tracksMeta), axis=1,
                                       result_type="expand")

                self.tracksSelf = tracksSelf_c.copy()
                # print(self.tracksSelf)
                self.save_to_csv(self.tracksSelf, path, record, currentId)
