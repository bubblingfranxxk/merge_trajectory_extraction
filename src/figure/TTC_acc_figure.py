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
    "recordingId": np.int32,
    "trackId": np.int32,
    "frame": np.int32,
    "trackLifetime": np.int32,
    "xCenter": np.float32,
    "yCenter": np.float32,
    "heading": np.float32,
    "width": np.float32,
    "length": np.float32,
    "xVelocity": np.float32,
    "yVelocity": np.float32,
    "xAcceleration": np.float32,
    "yAcceleration": np.float32,
    "lonVelocity": np.float32,
    "latVelocity": np.float32,
    "lonAcceleration": np.float32,
    "latAcceleration": np.float32,
    "traveledDistance": np.float32,
    "latLaneCenterOffset": np.float32,
    "laneletId": np.int32,
    "laneChange": bool,
    "lonLaneletPos": str,
    "leadId": np.int32,
    "rearId": np.int32,
    "leftRearId": np.int32,
    "leftLeadId": np.int32,
    "leftAlongsideId": str,
    "curxglobalutm": np.float32,
    "curyglobalutm": np.float32,
    "SurroundingVehiclesInfo": str,
    "RearVehicleId": np.int32,
    "RearDistance": np.float32,
    "RearDeltaV": np.float32,
    "RearDeltaAcceleration": np.float32,
    "RearTTCRaw1": np.float32,
    "RearTTCRaw2": np.float32,
    "RearTTCRaw3": np.float32,
    "LeadVehicleId": np.int32,
    "LeadDistance": np.float32,
    "LeadDeltaV": np.float32,
    "LeadDeltaAcceleration": np.float32,
    "LeadTTCRaw1": np.float32,
    "LeadTTCRaw2": np.float32,
    "LeadTTCRaw3": np.float32,
    "LeftRearVehicleId": np.int32,
    "LeftRearDistance": np.float32,
    "LeftRearDeltaV": np.float32,
    "LeftRearDeltaAcceleration": np.float32,
    "LeftRearTTCRaw1": np.float32,
    "LeftRearTTCRaw2": np.float32,
    "LeftRearTTCRaw3": np.float32,
    "LeftLeadVehicleId": np.int32,
    "LeftLeadDistance": np.float32,
    "LeftLeadDeltaV": np.float32,
    "LeftLeadDeltaAcceleration": np.float32,
    "LeftLeadTTCRaw1": np.float32,
    "LeftLeadTTCRaw2": np.float32,
    "LeftLeadTTCRaw3": np.float32,
    "LeftAlongsideVehicleId": np.int32,
    "LeftAlongsideDistance": np.float32,
    "LeftAlongsideDeltaV": np.float32,
    "LeftAlongsideDeltaAcceleration": np.float32,
    "LeftAlongsideTTCRaw1": np.float32,
    "LeftAlongsideTTCRaw2": np.float32,
    "LeftAlongsideTTCRaw3": np.float32,
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
    plt.savefig(path1, dpi=150, bbox_inches='tight')
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
    plt.savefig(path2, dpi=150, bbox_inches='tight')
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
    plt.savefig(path3, dpi=150, bbox_inches='tight')
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
    plt.savefig(path4, dpi=150, bbox_inches='tight')
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
    plt.savefig(path5, dpi=150, bbox_inches='tight')
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
