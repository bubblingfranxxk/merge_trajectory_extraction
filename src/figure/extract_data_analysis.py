# -*- coding = utf-8 -*-
# @Time : 2025/3/27 15:29
# @Author : 王砚轩
# @File : extract_data_analysis.py
# @Software: PyCharm
import os
import pandas as pd
from loguru import logger
from src.GAN.DataExtraction import value_range
import math
import matplotlib.pyplot as plt
import seaborn as sns
from src.GAN.data_normalization import recordingMapToLocation
import numpy as np
from utils.common import JS_div

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

fields_map = {
        'lonLaneletPos': 'lonPos',
        'latLaneCenterOffset': 'latPos',
        'heading': 'heading',
        'lonVelocity': 'velocity',
        'lonAcceleration': 'lonAcceleration',
        'latAcceleration': "latAcceleration"
    }


def process_data(extract_data_path, recording_data_path, output_path):
    # 获取 extract_data_path 下所有 csv 文件的完整路径
    csv_files = [os.path.join(extract_data_path, file)
                 for file in os.listdir(extract_data_path)
                 if file.endswith('.csv')]

    extract_data = []  # 用于存储最终提取的数据

    for file_name in csv_files:
        try:
            # 读取当前 CSV 文件，取第一行数据
            df = pd.read_csv(file_name)
            # 判断文件是否为空
            if df.empty:
                continue

            # 读取首行的 recordingId, trackId, MergingType (根据列名获取)
            recordingId = df.loc[0, 'recordingId']
            trackId = df.loc[0, 'trackId']
            mergingType = df.loc[0, 'MergingType']
        except Exception as e:
            logger.error(f"Error reading file {file_name}: {e}")
            continue

        # 根据 recordingId 构造对应的 tracksMeta 文件路径
        meta_filename = f"{recordingId}_tracksMeta.csv"
        meta_filepath = os.path.join(recording_data_path, meta_filename)

        class_value = ""  # 默认 class 字段为空

        if os.path.exists(meta_filepath):
            try:
                # 使用 pandas 读取 tracksMeta 文件
                meta_df = pd.read_csv(meta_filepath)
                # 筛选出 trackId 匹配的记录
                matched = meta_df[meta_df['trackId'] == trackId]
                if not matched.empty:
                    class_value = matched.iloc[0]['class']
            except Exception as e:
                logger.error(f"Error reading meta file {meta_filepath}: {e}")
        else:
            logger.warning(f"Warning: 文件 {meta_filepath} 不存在.")

        # 将提取的数据添加到列表中
        extract_data.append([recordingId, trackId, mergingType, class_value])

    # 将提取的数据转换为 DataFrame，并写入 CSV 文件
    output_df = pd.DataFrame(extract_data, columns=['recordingId', 'trackId', 'MergingType', 'class'])
    output_df.to_csv(output_path, index=False, encoding='utf-8')


def plot_mergingType_analysis(data_path, save_path=None):
    # 用于存储所有统计数据的列表
    stats = []

    # 字号设置
    title_fontsize = 12
    label_fontsize = 20
    tick_fontsize = 20
    legend_fontsize = 24
    annotation_fontsize = 18

    # 遍历目录下所有后缀为 extract_analysis.csv 的文件
    for file in os.listdir(data_path):
        if file.endswith("extract_analysis.csv"):
            # 提取文件名前面的数字 X，格式形如 "X_extract_analysis.csv"
            match = file.split('_')
            logger.debug(match)
            if match:
                x_value = float(match[0])
                logger.debug(x_value)
            else:
                print(f"无法从文件名 {file} 中提取数字X")
                continue

            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")
                continue

            # 统计当前文件中不同 MergingType 的数量
            count_series = df['MergingType'].value_counts()
            for merging_type, count in count_series.items():
                stats.append({'X': x_value, 'MergingType': merging_type, 'Count': count})

    if not stats:
        print("未找到有效的数据文件。")
        return

    # 汇总所有数据构造成 DataFrame
    stats_df = pd.DataFrame(stats)

    # 构造透视表: 行为 MergingType, 列为 X, 值为 Count, 缺失值填 0
    pivot_df = stats_df.pivot_table(index='MergingType', columns='X', values='Count', fill_value=0)

    # 绘制分组直方图
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), colormap="tab10", width=0.9)
    plt.xlabel("MergingType", fontsize=label_fontsize)
    plt.ylabel("Count", fontsize=label_fontsize)
    plt.title("不同 MergingType 与fTTC阈值的数据量统计", fontsize=title_fontsize)
    plt.legend(title="fTTC阈值", fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, rotation=0)
    plt.tight_layout()

    # 在每个柱状图顶部添加数据标签
    for patch in ax.patches:
        # 获取柱状图的高度
        height = patch.get_height()
        if height > 0:
            # 在柱子正上方添加数据标签，偏移5个像素
            ax.annotate(f'{int(height)}',
                        xy=(patch.get_x() + patch.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=annotation_fontsize)

    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.close()


def plot_class_analysis(data_path, save_path=None):
    # 用于存储所有统计数据的列表
    stats = []

    # 字号设置
    title_fontsize = 12
    label_fontsize = 20
    tick_fontsize = 20
    legend_fontsize = 24
    annotation_fontsize = 18

    # 遍历目录下所有后缀为 extract_analysis.csv 的文件
    for file in os.listdir(data_path):
        if file.endswith("extract_analysis.csv"):
            # 从文件名中提取 X，格式如 "X_extract_analysis.csv"，这里以 "_" 分割取第一个元素
            parts = file.split('_')
            if parts:
                try:
                    x_value = float(parts[0])
                except Exception as e:
                    logger.error(f"无法将 {parts[0]} 转换为数字: {e}")
                    continue
            else:
                logger.error(f"无法从文件名 {file} 中提取数字X")
                continue

            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"读取 {file_path} 失败: {e}")
                continue

            # 统计当前文件中不同 class 的数量
            count_series = df['class'].value_counts()
            for cls, count in count_series.items():
                stats.append({'X': x_value, 'class': cls, 'Count': count})

    if not stats:
        logger.error("未找到有效的数据文件。")
        return

    # 汇总所有数据构造成 DataFrame
    stats_df = pd.DataFrame(stats)

    # 构造透视表：行为 class，列为 X，值为 Count，缺失值填 0
    pivot_df = stats_df.pivot_table(index='class', columns='X', values='Count', fill_value=0)

    # 绘制分组直方图，使用 tab10 调色板，设置柱宽为 0.9
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), colormap="tab10", width=0.9)
    ax.set_xlabel("class", fontsize=label_fontsize)
    ax.set_ylabel("Count", fontsize=label_fontsize)
    ax.set_title("不同 fTTC 阈值下各 class 数据量统计", fontsize=title_fontsize)
    ax.legend(title="fTTC 阈值", fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, rotation=0)
    plt.tight_layout()

    # 在每个柱状图顶部添加数据标签
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(patch.get_x() + patch.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=annotation_fontsize)

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.close()


def plot_mergingType_class_heatmap(data_path, save_path=None):
    # 存储每个文件的统计结果及对应的 X 值
    heatmap_data = []

    # 遍历目录下所有后缀为 extract_analysis.csv 的文件
    for file in os.listdir(data_path):
        if file.endswith("extract_analysis.csv"):
            # 提取文件名前面的数字 X
            parts = file.split('_')
            if parts:
                try:
                    x_value = float(parts[0])
                except Exception as e:
                    logger.error(f"无法将 {parts[0]} 转换为数字: {e}")
                    continue
            else:
                logger.error(f"无法从文件名 {file} 中提取数字X")
                continue

            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"读取 {file_path} 失败: {e}")
                continue

            # 交叉统计 mergingType 和 class 的数量
            # 注意：这里假定 CSV 文件中同时存在 'MergingType' 与 'class' 列
            crosstab_df = pd.crosstab(df['MergingType'], df['class'])
            heatmap_data.append({'X': x_value, 'data': crosstab_df})

    if not heatmap_data:
        logger.error("未找到有效的数据文件。")
        return

    # 按照 X 值排序（可选）
    heatmap_data.sort(key=lambda x: x['X'])

    n_plots = len(heatmap_data)
    cols = 3
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    # 若只有一个子图，axes 可能不是二维数组
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # 循环绘制每个热力图
    for idx, item in enumerate(heatmap_data):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx][col_idx]

        # 使用 seaborn 绘制热力图
        sns.heatmap(item['data'], annot=True, fmt='d', cmap="YlOrRd", ax=ax, annot_kws={"fontsize": 20})
        ax.set_title(f"X = {item['X']}", fontsize=20)
        ax.set_xlabel("class", fontsize=12)
        ax.set_ylabel("MergingType", fontsize=12)
        # 设置坐标轴刻度字号调大
        ax.tick_params(axis='both', labelsize=18)

    # 清除空白子图（若存在）
    total_subplots = rows * cols
    if n_plots < total_subplots:
        for idx in range(n_plots, total_subplots):
            row_idx = idx // cols
            col_idx = idx % cols
            fig.delaxes(axes[row_idx][col_idx])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.close()


def process_and_plot(data_folder, save_path):
    # 读取 data_folder 下所有 csv 文件，并合并为一个 DataFrame
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    if not csv_files:
        print("未找到 CSV 文件！")
        return

    df_list = []
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            print(f"读取 {file} 失败: {e}")
    merged_df = pd.concat(df_list, ignore_index=True)

    # 添加 locationId 列，根据 recordingId 判断所属位置（要求录入的 recordingId 应该可转换为 int）
    def get_location(rec_id):
        try:
            rec_id = int(rec_id)
        except:
            return np.nan
        for loc, rec_list in recordingMapToLocation.items():
            if rec_id in rec_list:
                return loc
        return np.nan

    merged_df['locationId'] = merged_df['recordingId'].apply(get_location)
    # 仅保留 locationId 不为空的记录
    merged_df = merged_df.dropna(subset=['locationId'])

    # 定义需要统计的字段
    fields = ['lonLaneletPos', 'latLaneCenterOffset', 'heading', 'lonVelocity', 'lonAcceleration', 'latAcceleration']

    location_ids = ["2", "3", "5", "6"]  # 4 个分组

    # 创建图形，每个字段一行，共 6 行，每行 4 个子图
    n_rows = len(fields)
    n_cols = len(location_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)

    # 循环绘制每个字段在不同 locationId 下的分布图
    for i, field in enumerate(fields):
        for j, loc in enumerate(location_ids):
            ax = axes[i][j]
            # 筛选出当前 locationId 的数据
            sub_df = merged_df[merged_df['locationId'] == loc]
            data = sub_df[field].dropna()
            data = pd.to_numeric(data, errors='coerce').dropna()  # 转换为数值型
            if data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=24)
                continue

            # 绘制直方图（自动确定bin数，可根据需要调整 bins 参数）
            n, bins, patches = ax.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

            # 计算均值和标准差
            mean_val = data.mean()
            # std_val = data.std()
            # 绘制黑色竖虚线标注均值
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)

            # 添加图例，显示均值和方差
            ax.legend([f"mean={mean_val:.2f}, "
                       # f"std={std_val:.2f}"
                       ], fontsize=24)

            # 设置子图标题和坐标轴标签
            if j == 0:
                ax.set_ylabel(fields_map[field], fontsize=28)
            if i == 0:
                ax.set_title(f"locationId {loc}", fontsize=32)
            ax.tick_params(axis='both', labelsize=28)

    plt.tight_layout()
    # 保存图片
    plt.savefig(save_path, dpi=500)
    plt.close()


def process_and_plot_js_divergence(data_folder, save_path):
    """
    读取 data_folder 中所有 CSV 文件并合并，
    根据 recordingId 添加 locationId 列，
    针对每个字段计算不同 location 下数据分布的 JS 散度，
    并绘制 6 个热力图（2行3列），保存到 save_path。
    """
    # 这里选择用于计算 JS 散度的 locationId 分组
    location_ids = ["2", "3", "5", "6"]

    # 读取 data_folder 下所有 CSV 文件，并合并为一个 DataFrame
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    if not csv_files:
        print("未找到 CSV 文件！")
        return

    df_list = []
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            print(f"读取 {file} 失败: {e}")
    merged_df = pd.concat(df_list, ignore_index=True)

    # 根据 recordingId 添加 locationId 列（recordingId 应可转换为 int）
    def get_location(rec_id):
        try:
            rec_id = int(rec_id)
        except:
            return np.nan
        for loc, rec_list in recordingMapToLocation.items():
            if rec_id in rec_list:
                return loc
        return np.nan

    merged_df['locationId'] = merged_df['recordingId'].apply(get_location)
    # 仅保留 locationId 不为空的记录
    merged_df = merged_df.dropna(subset=['locationId'])

    # 定义需要计算 JS 散度的字段
    fields = ['lonLaneletPos', 'latLaneCenterOffset', 'heading',
              'lonVelocity', 'lonAcceleration', 'latAcceleration']
    num_bins = 50  # 可根据需要调整分箱数量

    # 创建图形，2行3列共6个子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, field in enumerate(fields):
        # 为当前字段构造一个 4x4 的 JS 散度矩阵
        n_loc = len(location_ids)
        js_matrix = np.zeros((n_loc, n_loc))
        # 先预处理全局数据，确保数值类型
        global_field = pd.to_numeric(merged_df[field], errors='coerce').dropna()
        if global_field.empty:
            print(f"字段 {field} 无有效数据")
            continue
        # 对每个 location_id 获取数据（转换为数值型）
        dist_dict = {}
        for loc in location_ids:
            sub_data = merged_df[merged_df['locationId'] == loc][field]
            sub_data = pd.to_numeric(sub_data, errors='coerce').dropna()
            dist_dict[loc] = sub_data.values

        # 计算每一对 location 之间的 JS 散度
        for i in range(n_loc):
            for j in range(n_loc):
                arr1 = dist_dict[location_ids[i]]
                arr2 = dist_dict[location_ids[j]]
                # 若任一组数据为空，则置为 NaN
                if arr1.size == 0 or arr2.size == 0:
                    js_matrix[i, j] = np.nan
                else:
                    js_matrix[i, j] = JS_div(arr1, arr2, num_bins)

        # 绘制当前字段的 JS 散度热力图
        ax = axes[idx]
        # 绘制当前字段的 JS 散度热力图，共用比例尺 vmin=0, vmax=1
        ax = axes[idx]
        sns.heatmap(js_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=location_ids, yticklabels=location_ids, ax=ax,
                    vmin=0, vmax=1, annot_kws={"fontsize": 20})
        ax.set_title(f"JS Divergence for {fields_map[field]}", fontsize=24)
        ax.set_xlabel("locationId", fontsize=28)
        ax.set_ylabel("locationId", fontsize=28)
        ax.tick_params(axis='both', labelsize=28)

    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()


if __name__ == '__main__':
    # 配置参数
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"  # 合并后的输出文件
    dataPath = rootPath + "/drone-dataset-tools-master/data/"  # CSV文件所在文件夹
    extract_folder = assetPath + '/extracted_data/'
    extract_lead_folder = assetPath + '/adjusted_surrounding/leadId/'
    extract_rear_folder = assetPath + '/adjusted_surrounding/rearId/'

    normalized_folder = assetPath + '/normalized_data/'
    normalized_lead_folder = assetPath + '/normalization_surrounding/leadId/'
    normalized_rear_folder = assetPath + '/normalization_surrounding/rearId/'


    # 绘制不同TTC阈值提取得到的汇入场景 class和mergingType数据
    process_data(extract_folder, dataPath, assetPath + f"{value_range[1]}_extract_analysis.csv")

    # 绘制不同Location的不同mergingType分布图
    plot_mergingType_analysis(assetPath, assetPath + f"extract_analysis_mergingtype.png")

    # # 绘制不同Locaiton的不同车型class的分布图
    # plot_class_analysis(assetPath, assetPath + f"extract_analysis_class.png")

    # # 绘制不同Location的 mergingType-Class的交叉分类热力图
    # plot_mergingType_class_heatmap(assetPath, assetPath + f"extract_analysis_heatmap.png")

    # # 绘制不同Location 的变量分布图
    # process_and_plot(extract_folder, assetPath + f"extract_distribution.png")
    # process_and_plot(extract_lead_folder, assetPath + f"extractLead_distribution.png")
    # process_and_plot(extract_rear_folder, assetPath + f"extractRear_distribution.png")

    # 绘制不同Location的变量间JS散度heatmap
    # process_and_plot_js_divergence(extract_folder, assetPath + f"extract_JS.png")
    # process_and_plot_js_divergence(extract_lead_folder, assetPath + f"extractLead_JS.png")
    # process_and_plot_js_divergence(extract_rear_folder, assetPath + f"extractRear_JS.png")

    # 绘制标准化后的分布图和JS散度图
    process_and_plot(normalized_folder, assetPath + f"normalized_distribution.png")
    process_and_plot(normalized_lead_folder, assetPath + f"normalizedLead_distribution.png")
    process_and_plot(normalized_rear_folder, assetPath + f"normalizedRear_distribution.png")

    process_and_plot_js_divergence(normalized_folder, assetPath + f"normalized_JS.png")
    process_and_plot_js_divergence(normalized_lead_folder, assetPath + f"normalizedLead_JS.png")
    process_and_plot_js_divergence(normalized_rear_folder, assetPath + f"normalizedRear_JS.png")


