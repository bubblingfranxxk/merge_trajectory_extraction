import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from loguru import logger
from scipy.optimize import fsolve
import scipy


# JS散度公式
def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M, base=2) + 0.5 * scipy.stats.entropy(q, M, base=2)


def JS_div(arr1, arr2, num_bins, min0, max0):
    arr1 += 1e-10
    arr2 += 1e-10
    bins = np.linspace(min0, max0, num=num_bins)
    PDF1 = pd.cut(arr1, bins, duplicates="drop").value_counts() / len(arr1)
    PDF2 = pd.cut(arr2, bins, duplicates="drop").value_counts() / len(arr2)
    #    return min(JS_divergence(PDF1.values,PDF2.values),1)
    return JS_divergence(PDF1.values, PDF2.values)


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
                        plt.tex_t(data.index[k - 1], data['latLaneCenterOffset'].iloc[k - 1], str(annotation_counter),
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


def ellipses_tangent_time(a1, b1, theta1, x1, y1, vx, vy, a2, b2, theta2, x2, y2, guess_t):
    # 计算静止椭圆的一般形式方程系数
    A1, B1, C1, D1, E1, F1 = ellipse_general_form(a1, b1, theta1, x1, y1)

    def tangent_equation(t):
        # 计算移动椭圆在当前时刻的中心坐标
        x_T = x2 + vx * t
        y_T = y2 + vy * t

        # 计算移动椭圆的一般形式方程系数
        A2, B2, C2, D2, E2, F2 = ellipse_general_form(a2, b2, theta2, x_T, y_T)

        # 构建方程组
        def equations(Vars):
            x, y = Vars
            eq1 = A1 * x ** 2 + B1 * x * y + C1 * y ** 2 + D1 * x + E1 * y + F1
            eq2 = A2 * x ** 2 + B2 * x * y + C2 * y ** 2 + D2 * x + E2 * y + F2
            return [eq1, eq2]

        # 初始猜测点
        guess = [(x1 + x_T) / 2.0, (y1 + y_T) / 2.0]

        # 解方程组
        solution = fsolve(equations, guess, fprime=None, col_deriv=False, maxfev=1000)

        x, y = solution
        eq1_value = A1 * x ** 2 + B1 * x * y + C1 * y ** 2 + D1 * x + E1 * y + F1
        eq2_value = A2 * x ** 2 + B2 * x * y + C2 * y ** 2 + D2 * x + E2 * y + F2

        return np.abs(eq1_value) + np.abs(eq2_value)

    # 找到相切时刻
    t_solutions = fsolve(tangent_equation, guess_t, fprime=None, col_deriv=False, maxfev=1000)
    temp = min((x for x in t_solutions if x > 0), default=None)
    if temp is None:
        t_solution = t_solutions[0]
    else:
        t_solution = temp

    # 检查解是否有效
    if t_solution is not None:
        # logger.debug(f"t_solution is {t_solution}")
        x_t = x2 + vx * t_solution
        y_t = y2 + vy * t_solution
        return t_solution, x_t, y_t
    else:
        return None, None, None


def sigmoid_gradient(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)


def reload_ellipse_general_form(a, b, theta, x0, y0):
    # 将角度转换为弧度
    theta_rad = np.deg2rad(theta)

    # 计算系数
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    A = (cos_theta / a) ** 2 + (sin_theta / b) ** 2
    B = 2 * (sin_theta * cos_theta) * (1 / a ** 2 - 1 / b ** 2)
    C = (sin_theta / a) ** 2 + (cos_theta / b) ** 2
    D = -2 * A * x0 - B * y0
    E = -B * x0 - 2 * C * y0
    F = A * x0 ** 2 + B * x0 * y0 + C * y0 ** 2 - 1

    return A, B, C, D, E, F


def reload_ellipses_tangent_time(a1, b1, theta1, x1, y1, vx, vy, a2, b2, theta2, x2, y2):
    # 计算静止椭圆的一般形式方程系数
    A1, B1, C1, D1, E1, F1 = ellipse_general_form(a1, b1, theta1, x1, y1)

    def tangent_equation(t):
        x_T = x2 + vx * t
        y_T = y2 + vy * t

        # 计算移动椭圆的一般形式方程系数
        A2, B2, C2, D2, E2, F2 = ellipse_general_form(a2, b2, theta2, x_T, y_T)

        # 定义相切条件
        def equations(vars):
            x, y = vars
            eq1 = A1 * x ** 2 + B1 * x * y + C1 * y ** 2 + D1 * x + E1 * y + F1
            eq2 = A2 * x ** 2 + B2 * x * y + C2 * y ** 2 + D2 * x + E2 * y + F2
            return [eq1, eq2]

        # 初始猜测点
        guess = [(x1 + x_T) / 2.0, (y1 + y_T) / 2.0]

        # 解方程组
        solution = fsolve(equations, guess)

        x, y = solution
        eq1_value = A1 * x ** 2 + B1 * x * y + C1 * y ** 2 + D1 * x + E1 * y + F1
        eq2_value = A2 * x ** 2 + B2 * x * y + C2 * y ** 2 + D2 * x + E2 * y + F2

        # 如果两个方程的值都接近0，则认为两个椭圆相切
        return np.abs(eq1_value) + np.abs(eq2_value)

    # 尝试不同的初始猜测值
    t_values = np.linspace(10, 50, 100)  # 从0到10，取100个值作为初始猜测值
    best_t = None
    best_value = np.inf

    for t_guess in t_values:
        try:
            t_solution = fsolve(tangent_equation, t_guess, maxfev=1000)[0]
            xt = x2 + vx * t_solution
            yt = y2 + vy * t_solution
            value = tangent_equation(t_solution)
            if value < best_value:
                best_value = value
                best_t = t_solution
        except:
            continue

    # 如果找到最优解，计算相切时运动椭圆的中心点坐标
    if best_t is not None:
        xt = x2 + vx * best_t
        yt = y2 + vy * best_t
        return best_t, xt, yt
    else:
        return None, None, None
