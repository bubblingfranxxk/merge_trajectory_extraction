import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from loguru import logger
from scipy.optimize import fsolve


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

        # 添加图例
        plt.legend()

        # 添加标题和标签
        plt.title(f'Batch {i + 1} - {start_idx + 1} to {end_idx}')
        plt.xlabel('Standardized Index')
        plt.ylabel('Y-axis')

        # 保存图形到文件
        plt.savefig(f'{save_path}/{file_prefix}_{i + 1}.png')

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

        # 添加图例
        plt.legend()

        # 添加标题和标签
        plt.title(f'Remaining Data - Batch {num_batches + 1} - {start_idx + 1} to {end_idx}')
        plt.xlabel('Standardized Index')
        plt.ylabel('Y-axis')

        # 保存图形到文件
        plt.savefig(f'{save_path}/{file_prefix}_{num_batches + 1}.png')

        # 关闭图形
        plt.close()


def ellipse_general_form(a, b, theta, x0, y0):
    # 将角度转换为弧度
    theta_rad = np.deg2rad(theta)

    # 调整长短轴长度，确保长轴大于或等于短轴
    a_adjusted = max(a, b)
    b_adjusted = min(a, b)

    # 将椭圆旋转至长轴与 x 轴对齐
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # 将旋转后的标准椭圆方程展开并整理成一般形式的二次方程
    A = (cos_theta / a_adjusted) ** 2 + (sin_theta / b_adjusted) ** 2
    B = 2 * ((sin_theta * cos_theta) / (a_adjusted ** 2) - (sin_theta * cos_theta) / (b_adjusted ** 2))
    C = (sin_theta / a_adjusted) ** 2 + (cos_theta / b_adjusted) ** 2
    D = -2 * A * x0 - B * y0
    E = -B * x0 - 2 * C * y0
    F = A * x0 ** 2 + B * x0 * y0 + C * y0 ** 2 - 1

    return A, B, C, D, E, F


def ellipses_tangent_time(a1, b1, theta1, x1, y1, vx, vy, a2, b2, theta2, x2, y2):
    def tangent_equation(t):
        # 计算移动椭圆在当前时刻的中心坐标
        x_t = x2 + vx * t
        y_t = y2 + vy * t

        # 计算两个椭圆的一般形式方程系数
        A1, B1, C1, D1, E1, F1 = ellipse_general_form(a1, b1, theta1, x1, y1)
        A2, B2, C2, D2, E2, F2 = ellipse_general_form(a2, b2, theta2, x_t, y_t)

        # 定义相切方程，即两个椭圆的距离为0
        return (A1 - A2) * x_t ** 2 + (B1 - B2) * x_t * y_t + (C1 - C2) * y_t ** 2 + (D1 - D2) * x_t + (E1 - E2) * y_t \
            + (F1 - F2)

    # 求解相切方程的根，即相切时刻
    t_solution = fsolve(tangent_equation, 0, fprime=sigmoid_gradient, col_deriv=False, maxfev=200)

    # 检查解是否有效
    if len(t_solution) > 0:
        t_solution = t_solution[0]
        xt = x2 + vx * t_solution
        yt = y2 + vy * t_solution
        return t_solution, xt, yt
    else:
        return None, None, None


def sigmoid_gradient(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)
