# -*- coding = utf-8 -*-
# @Time : 2024/5/28 10:47
# @Author : 王砚轩
# @File : TTC_acc_figure.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from loguru import logger
import shutil

dtype_spec = {
    "recordingId": int,
    "trackId": int,
    "frame": int,
    "trackLifetime": int,
    "xCenter": float,
    "yCenter": float,
    "heading": float,
    "width": float,
    "length": float,
    "xVelocity": float,
    "yVelocity": float,
    "xAcceleration": float,
    "yAcceleration": float,
    "lonVelocity": float,
    "latVelocity": float,
    "lonAcceleration": float,
    "latAcceleration": float,
    "traveledDistance": float,
    "latLaneCenterOffset": float,
    "laneletId": int,
    "laneChange": bool,
    "lonLaneletPos": str,
    "leadId": int,
    "rearId": int,
    "leftRearId": int,
    "leftLeadId": int,
    "leftAlongsideId": str,
    "curxglobalutm": float,
    "curyglobalutm": float,
    "SurroundingVehiclesInfo": str,
    "RearVehicleId": int,
    "RearDistance": float,
    "RearDeltaV": float,
    "RearDeltaAcceleration": float,
    "RearTTCRaw1": float,
    "RearTTCRaw2": float,
    "RearTTCRaw3": float,
    "LeadVehicleId": int,
    "LeadDistance": float,
    "LeadDeltaV": float,
    "LeadDeltaAcceleration": float,
    "LeadTTCRaw1": float,
    "LeadTTCRaw2": float,
    "LeadTTCRaw3": float,
    "LeftRearVehicleId": int,
    "LeftRearDistance": float,
    "LeftRearDeltaV": float,
    "LeftRearDeltaAcceleration": float,
    "LeftRearTTCRaw1": float,
    "LeftRearTTCRaw2": float,
    "LeftRearTTCRaw3": float,
    "LeftLeadVehicleId": int,
    "LeftLeadDistance": float,
    "LeftLeadDeltaV": float,
    "LeftLeadDeltaAcceleration": float,
    "LeftLeadTTCRaw1": float,
    "LeftLeadTTCRaw2": float,
    "LeftLeadTTCRaw3": float,
    "LeftAlongsideVehicleId": int,
    "LeftAlongsideDistance": float,
    "LeftAlongsideDeltaV": float,
    "LeftAlongsideDeltaAcceleration": float,
    "LeftAlongsideTTCRaw1": float,
    "LeftAlongsideTTCRaw2": float,
    "LeftAlongsideTTCRaw3": float,
    "MergingType": str
}


def setFigure(trajset, path1, path2=None, path3=None, path4=None, path5=None):
    n = len(trajset)

    # 创建一个3x3的子图布局
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(50, 15))

    # 使用axes.flat遍历并绘制每个子图
    for i, ax in enumerate(axes.flat):
        if i == n:
            break
        temp = trajset[i]
        xmin, xmax = min(temp['frame']), max(temp['frame'])
        x = temp[temp['RearTTCRaw1'] != 999]['frame']
        y1 = temp[temp['RearTTCRaw1'] != 999]['RearTTCRaw1']
        y2 = temp[temp['RearTTCRaw1'] != 999]['RearTTCRaw2']
        y3 = temp[temp['RearTTCRaw1'] != 999]['RearTTCRaw3']
        y4 = temp[temp['RearTTCRaw1'] != 999]['lonAcceleration']
        recordingId = temp['recordingId'].values[0]
        trackId = temp['trackId'].values[0]
        string = "recording " + str(recordingId) + " track " + str(trackId)

        # 绘制第一组图
        ax.plot(x, y1, linestyle='-', color='b', label='TTC')
        ax.plot(x, y2, linestyle='-', color='r', label='pTTC')
        ax.plot(x, y3, linestyle='-', color='g', label='bTTC')
        ax.set_xlim(xmin, xmax)

        # 创建第二个y轴，共享x轴
        ax2 = ax.twinx()
        ax2.plot(x, y4, linestyle='-', color='m', label='xAcceleration')

        # 设置
        ax.set_ylabel('s')
        ax2.set_ylabel('m/s2')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_title(string+' Rear')

    # 调整布局，使子图不重叠
    plt.tight_layout()

    # 保存
    plt.savefig(path1, dpi=80, bbox_inches='tight')
    plt.close()

    # 创建一个3x3的子图布局
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(50, 15))

    # 使用axes.flat遍历并绘制每个子图
    for i, ax in enumerate(axes.flat):
        if i == n:
            break
        temp = trajset[i]
        xmin, xmax = min(temp['frame']), max(temp['frame'])
        x = temp[temp['LeadTTCRaw1'] != 999]['frame']
        y1 = temp[temp['LeadTTCRaw1'] != 999]['LeadTTCRaw1']
        y2 = temp[temp['LeadTTCRaw1'] != 999]['LeadTTCRaw2']
        y3 = temp[temp['LeadTTCRaw1'] != 999]['LeadTTCRaw3']
        y4 = temp[temp['LeadTTCRaw1'] != 999]['lonAcceleration']
        recordingId = temp['recordingId'].values[0]
        trackId = temp['trackId'].values[0]
        string = "recording " + str(recordingId) + " track " + str(trackId)

        # 绘制第一组图
        ax.plot(x, y1, linestyle='-', color='b', label='TTC')
        ax.plot(x, y2, linestyle='-', color='r', label='pTTC')
        ax.plot(x, y3, linestyle='-', color='g', label='bTTC')
        ax.set_xlim(xmin, xmax)

        # 创建第二个y轴，共享x轴
        ax2 = ax.twinx()
        ax2.plot(x, y4, linestyle='-', color='m', label='xAcceleration')

        # 设置
        ax.set_ylabel('s')
        ax2.set_ylabel('m/s2')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_title(string + ' Lead')

    # 调整布局，使子图不重叠
    plt.tight_layout()

    # 保存
    plt.savefig(path2, dpi=80, bbox_inches='tight')
    plt.close()

    # 创建一个3x3的子图布局
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(50, 15))

    # 使用axes.flat遍历并绘制每个子图
    for i, ax in enumerate(axes.flat):
        if i == n:
            break
        temp = trajset[i]
        xmin, xmax = min(temp['frame']), max(temp['frame'])
        x = temp[temp['LeftRearTTCRaw1'] != 999]['frame']
        y1 = temp[temp['LeftRearTTCRaw1'] != 999]['LeftRearTTCRaw1']
        y2 = temp[temp['LeftRearTTCRaw1'] != 999]['LeftRearTTCRaw2']
        y3 = temp[temp['LeftRearTTCRaw1'] != 999]['LeftRearTTCRaw3']
        y4 = temp[temp['LeftRearTTCRaw1'] != 999]['lonAcceleration']
        recordingId = temp['recordingId'].values[0]
        trackId = temp['trackId'].values[0]
        string = "recording " + str(recordingId) + " track " + str(trackId)

        # 绘制第一组图
        ax.plot(x, y1, linestyle='-', color='b', label='TTC')
        ax.plot(x, y2, linestyle='-', color='r', label='pTTC')
        ax.plot(x, y3, linestyle='-', color='g', label='bTTC')
        ax.set_xlim(xmin, xmax)

        # 创建第二个y轴，共享x轴
        ax2 = ax.twinx()
        ax2.plot(x, y4, linestyle='-', color='m', label='xAcceleration')

        # 设置
        ax.set_ylabel('s')
        ax2.set_ylabel('m/s2')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_title(string + ' LeftRear')

    # 调整布局，使子图不重叠
    plt.tight_layout()

    # 保存
    plt.savefig(path3, dpi=80, bbox_inches='tight')
    plt.close()

    # 创建一个3x3的子图布局
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(50, 15))

    # 使用axes.flat遍历并绘制每个子图
    for i, ax in enumerate(axes.flat):
        if i == n:
            break
        temp = trajset[i]
        xmin, xmax = min(temp['frame']), max(temp['frame'])
        x = temp[temp['LeftLeadTTCRaw1'] != 999]['frame']
        y1 = temp[temp['LeftLeadTTCRaw1'] != 999]['LeftLeadTTCRaw1']
        y2 = temp[temp['LeftLeadTTCRaw1'] != 999]['LeftLeadTTCRaw2']
        y3 = temp[temp['LeftLeadTTCRaw1'] != 999]['LeftLeadTTCRaw3']
        y4 = temp[temp['LeftLeadTTCRaw1'] != 999]['lonAcceleration']
        recordingId = temp['recordingId'].values[0]
        trackId = temp['trackId'].values[0]
        string = "recording " + str(recordingId) + " track " + str(trackId)

        # 绘制第一组图
        ax.plot(x, y1, linestyle='-', color='b', label='TTC')
        ax.plot(x, y2, linestyle='-', color='r', label='pTTC')
        ax.plot(x, y3, linestyle='-', color='g', label='bTTC')
        ax.set_xlim(xmin, xmax)

        # 创建第二个y轴，共享x轴
        ax2 = ax.twinx()
        ax2.plot(x, y4, linestyle='-', color='m', label='xAcceleration')

        # 设置
        ax.set_ylabel('s')
        ax2.set_ylabel('m/s2')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_title(string + ' LeftLead')

    # 调整布局，使子图不重叠
    plt.tight_layout()

    # 保存
    plt.savefig(path4, dpi=80, bbox_inches='tight')
    plt.close()

    # 创建一个3x3的子图布局
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(50, 15))

    # 使用axes.flat遍历并绘制每个子图
    for i, ax in enumerate(axes.flat):
        if i == n:
            break
        temp = trajset[i]
        xmin, xmax = min(temp['frame']), max(temp['frame'])
        x = temp[temp['LeftAlongsideTTCRaw1'] != 999]['frame']
        y1 = temp[temp['LeftAlongsideTTCRaw1'] != 999]['LeftAlongsideTTCRaw1']
        y2 = temp[temp['LeftAlongsideTTCRaw1'] != 999]['LeftAlongsideTTCRaw2']
        y3 = temp[temp['LeftAlongsideTTCRaw1'] != 999]['LeftAlongsideTTCRaw3']
        y4 = temp[temp['LeftAlongsideTTCRaw1'] != 999]['lonAcceleration']
        recordingId = temp['recordingId'].values[0]
        trackId = temp['trackId'].values[0]
        string = "recording " + str(recordingId) + " track " + str(trackId)

        # 绘制第一组图
        ax.plot(x, y1, linestyle='-', color='b', label='TTC')
        ax.plot(x, y2, linestyle='-', color='r', label='pTTC')
        ax.plot(x, y3, linestyle='-', color='g', label='bTTC')
        ax.set_xlim(xmin, xmax)

        # 创建第二个y轴，共享x轴
        ax2 = ax.twinx()
        ax2.plot(x, y4, linestyle='-', color='m', label='xAcceleration')

        # 设置
        ax.set_ylabel('s')
        ax2.set_ylabel('m/s2')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_title(string + ' LeftAlongside')

    # 调整布局，使子图不重叠
    plt.tight_layout()

    # 保存
    plt.savefig(path5, dpi=80, bbox_inches='tight')
    plt.close()


def create_output_folder(path, folder):
    output_path = os.path.join(path, folder)

    # 检查output文件夹是否已经存在
    if os.path.exists(output_path):
        # 如果已存在，则删除
        shutil.rmtree(output_path)

    # 创建output文件夹
    os.makedirs(output_path)


def main():
    rootPath = os.path.abspath('../../')
    # 这里使用合共的meringDataNew
    assetPath = rootPath + "/asset/"
    # 获取文件夹中合并的MergingTrajectory
    traj_files = [f for f in os.listdir(assetPath) if f.endswith('.csv')]
    create_output_folder(assetPath, 'TTC_compare')
    TTCpath = assetPath + "/TTC_compare/"
    create_output_folder(TTCpath, "Rear")
    create_output_folder(TTCpath, "Lead")
    create_output_folder(TTCpath, "LeftRear")
    create_output_folder(TTCpath, "LeftLead")
    create_output_folder(TTCpath, "LeftAlongside")

    for file in traj_files:
        if "Trajectory" not in file:
            continue
        file_path = assetPath + file
        trajectory = pd.read_csv(file_path, dtype=dtype_spec)
        logger.info("Loading {}", file)

        singleTraj = pd.DataFrame()
        listTraj = []
        count_plt = 0
        num = 1
        for index, row in trajectory.iterrows():
            if count_plt == 9:
                setFigure(trajset=listTraj,
                          path1=TTCpath+'/Rear/'+str(num)+'.png',
                          path2=TTCpath+'/Lead/'+str(num)+'.png',
                          path3=TTCpath+'/LeftRear/'+str(num)+'.png',
                          path4=TTCpath+'/LeftLead/'+str(num)+'.png',
                          path5=TTCpath+'/LeftAlongside/'+str(num)+'.png')
                listTraj = []
                num += 1
                logger.info("Figure is Printed. Recording is {}, track is {}.", row['recordingId'], row['trackId'])
                count_plt = 0
            if not singleTraj.empty \
                    and (row['recordingId'] != singleTraj['recordingId'].values[0]
                         or row['trackId'] != singleTraj['trackId'].values[0]):
                listTraj.append(singleTraj)
                singleTraj = pd.DataFrame()
                count_plt += 1

            newrow = pd.DataFrame(row).transpose()
            # print(newrow)
            singleTraj = pd.concat([singleTraj, newrow])


if __name__ == '__main__':
    main()
