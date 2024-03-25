import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from loguru import logger


# 变道筛选/过滤 把两次压线间隔小于1s的变道删掉前者，看作1次变道
def filterLaneChange(curLcIndex):
    gaplist = []
    tepIndex = []
    if len(curLcIndex) == 1:
        gaplist.append(-1)
    else:
        for i in range(len(curLcIndex) - 1):
            gap = round((curLcIndex[i + 1] - curLcIndex[i]) / 25, 2)
            gaplist.append(gap)
        for j in range(len(gaplist)):
            if gaplist[j] < 1:
                tepIndex.append(j)
            else:
                continue
        curLcIndex = np.delete(curLcIndex, tepIndex)

    return curLcIndex


def calculateLonVelocity(row, direction, variable):
    angle = math.radians(row["heading"])
    cosvalue = math.cos(angle)
    sinvalue = math.sin(angle)
    first = "x" + variable
    second = "y" + variable

    if direction == "lon":
        return row[first] * cosvalue + row[second] * sinvalue
    elif direction == "lat":
        return row[first] * sinvalue - row[second] * cosvalue


def processLaneletData(row, type):
    if type == "int":
        if ";" not in str(row):
            return int(row)
        else:
            return int(row.split(";")[0])
    elif type == "float":
        if ";" not in str(row):
            return float(row)
        else:
            return float(row.split(";")[0])


def standardize_index(df):
    scaler = StandardScaler()
    df.index = scaler.fit_transform(df.index.to_numpy().reshape(-1, 1)).flatten()
    return df


def plot_data_in_batches(data_sets, batch_size, file_prefix="batch", save_path="."):
    num_batches = len(data_sets) // batch_size
    remainder = len(data_sets) % batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = data_sets[start_idx:end_idx]

        # 创建图形
        plt.figure()

        # 绘制折线图
        annotation_counter = 1
        for j, data in enumerate(batch_data):
            data = standardize_index(data)
            plt.plot(data.index, data['latLaneCenterOffset'],
                     label="track id {}".format(data['trackId'].iloc[0]))

            # 标注变化点
            for k in range(1, len(data)):
                if (data['latLaneCenterOffset'].iloc[k] * data['latLaneCenterOffset'].iloc[k - 1] < 0) and \
                        abs(data['latLaneCenterOffset'].iloc[k] - data['latLaneCenterOffset'].iloc[k - 1]) > 0.5:
                    # 使用空心圆环并设置大小
                    plt.scatter(data.index[k - 1], data['latLaneCenterOffset'].iloc[k - 1], color='red', marker='o',
                                facecolors='none', s=100)

                    # 标注点的文本信息
                    if annotation_counter >= batch_size:
                        plt.text(data.index[k - 1], data['latLaneCenterOffset'].iloc[k - 1], str(annotation_counter),
                                 color='black', ha='center', va='center')
                    annotation_counter += 1  # 更新计数器

        # 设置图例位置
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # 添加标题和标签
        plt.title(f'Batch {i+1} - {start_idx+1} to {end_idx}')
        plt.xlabel('Standardized Index')
        plt.ylabel('laneoffset')    # 设置y轴标题

        # 保存图形到文件
        plt.savefig(f'{save_path}/{file_prefix}_{i+1}.png', bbox_inches='tight')

        # 关闭图形
        plt.close()

    # 处理余下的数据
    if remainder > 0:
        start_idx = num_batches * batch_size
        end_idx = len(data_sets)
        batch_data = data_sets[start_idx:end_idx]

        # 创建图形
        plt.figure()

        # 绘制折线图
        annotation_counter = 1
        for j, data in enumerate(batch_data):
            data = standardize_index(data)
            plt.plot(data.index, data['latLaneCenterOffset'],
                     label="track id {}".format(data['trackId'].iloc[0]))

            # 将 Series 转换为 DataFrame
            data = pd.DataFrame(data)

            # 标注变化点
            for k in range(1, len(data)):
                if (data['latLaneCenterOffset'].iloc[k] * data['latLaneCenterOffset'].iloc[k - 1] < 0) and \
                        abs(data['latLaneCenterOffset'].iloc[k] - data['latLaneCenterOffset'].iloc[k - 1]) > 0.5:
                    # 使用空心圆环并设置大小
                    plt.scatter(data.index[k - 1], data['latLaneCenterOffset'].iloc[k - 1], color='red', marker='o',
                                facecolors='none', s=100)

                    # 标注点的文本信息
                    if annotation_counter >= remainder:
                        plt.text(data.index[k - 1], data['latLaneCenterOffset'].iloc[k - 1], str(annotation_counter),
                                 color='black', ha='center', va='center')
                    annotation_counter += 1  # 更新计数器

        # 设置图例位置
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # 添加标题和标签
        plt.title(f'Remaining Data - Batch {num_batches+1} - {start_idx+1} to {end_idx}')
        plt.xlabel('Standardized Index')
        plt.ylabel('laneoffset')    # 设置y轴标题

        # 保存图形到文件
        plt.savefig(f'{save_path}/{file_prefix}_{num_batches+1}.png', bbox_inches='tight')

        # 关闭图形
        plt.close()
