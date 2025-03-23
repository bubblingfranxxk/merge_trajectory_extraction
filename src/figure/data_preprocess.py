# -*- coding = utf-8 -*-
# @Time : 2025/3/13 15:13
# @Author : 王砚轩
# @File : data_preprocess.py
# @Software: PyCharm

import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from src.GAN.data_normalization import recordingMapToLocation
import matplotlib.gridspec as gridspec
from scipy import stats
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei'] 等支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def merge_and_analyze_1(input_folder, output_file, suffix=".csv", analysis_prefix="analysis"):
    """
    完整数据处理流程：合并CSV并执行分析

    参数:
    input_folder: 输入文件夹路径
    output_file: 合并后的CSV路径
    suffix: 目标文件后缀
    analysis_prefix: 分析结果文件前缀

    返回:
    dict: 包含合并与分析结果的字典
    """
    # 合并CSV文件
    merge_result = merge_csv_files(input_folder, output_file, suffix)

    if "错误" in merge_result or "error" in merge_result.lower():
        return {"merge_status": "failed", "analysis_status": "skipped", "message": merge_result}

    # 执行数据分析
    analysis_result = analyze_location_records(output_file, f"{assetPath}{analysis_prefix}")

    # 执行可视化（要求合并后的数据中包含 recordingId 字段）
    vis_result = visualize_duration_distribution(output_file, output_image_path=f"{assetPath}{analysis_prefix}"
                                                                                f"_duration_hist.png")

    return {
        "merge_status": "success",
        "analysis_status": "success" if "error" not in analysis_result else "failed",
        "visualization_status": "success" if "错误" not in vis_result else "failed",
        "merge_output": output_file,
        "analysis_outputs": analysis_result,
        "visualization_output": vis_result
    }


def merge_and_analyze_2(input_folder, output_file, suffix=".csv", analysis_prefix="analysis"):
    # 合并CSV文件
    merge_result = merge_csv_files(input_folder, output_file, suffix,
                                   select_columns=['recordingId', 'trackId', 'class'])

    if "错误" in merge_result or "error" in merge_result.lower():
        return {"merge_status": "failed", "analysis_status": "skipped", "message": merge_result}

    plot_stacked_histogram(output_file, f"{assetPath}{analysis_prefix}_vehilce_class")

    return {
        "merge_status": "success",
        "merge_output": output_file
    }


def merge_and_analyze_3(input_folder, output_file, suffix=".csv", analysis_prefix="analysis"):
    # 合并CSV文件
    # merge_result = merge_csv_files(input_folder, output_file, suffix,
    #                                select_columns=['recordingId', 'trackId', 'lonVelocity'])
    #
    # if "错误" in merge_result or "error" in merge_result.lower():
    #     return {"merge_status": "failed", "analysis_status": "skipped", "message": merge_result}
    test_speed_limit_effect(output_file, f"{assetPath}{analysis_prefix}_velocity_boxplot")
    fig = plot_lon_velocity_distribution(output_file, f"{assetPath}{analysis_prefix}_velocity_distribution")

    return {
        "merge_status": "success",
        "plot_status": fig,
        "merge_output": output_file
    }


def test_speed_limit_effect(csv_file, save_path=None):
    """
    读取CSV数据，并检验限速措施是否对车速有影响：
      - locationId为0和1的数据限速为100 km/h
      - locationId为4的数据限速为120 km/h
      - 其他locationId无速限

    该函数将：
      1. 分组：创建100 km/h组、120 km/h组和无速限组
      2. 使用独立样本t检验比较每个限速组与无速限组的均值是否存在显著差异
      3. 使用单因素方差分析(ANOVA)比较三个组间的均值差异
      4. 计算并输出各限速组中超出限速的比例

    参数:
        csv_file (str): CSV文件路径。文件应包含列名['recordingId', 'locationId', 'trackId', 'lonVelocity']。
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    # 删除 lonVelocity 为负值的记录
    df = df[df['lonVelocity'] > 0]
    # 应用映射函数生成 locationId 列
    df["locationId"] = df["recordingId"].apply(map_recording_to_location)
    # 将 locationId 转为字符串，以便排序和作为分类变量
    df["locationId"] = df["locationId"].astype('int32')

    # 分组：
    group_100 = df[df['locationId'].isin([0, 1])]['lonVelocity']
    # logger.debug(group_100)
    group_120 = df[df['locationId'] == 4]['lonVelocity']
    group_no_limit = df[~df['locationId'].isin([0, 1, 4])]['lonVelocity']

    # 输出每个分组的样本量，便于调试
    logger.info(f"100 km/h 限速组样本量: {len(group_100)}")
    logger.info(f"120 km/h 限速组样本量: {len(group_120)}")
    logger.info(f"无速限组样本量: {len(group_no_limit)}")

    # t检验及效应量计算
    # 定义一个计算Cohen's d的函数（适用于独立样本t检验）
    def cohen_d(x, y):
        n1, n2 = len(x), len(y)
        mean1, mean2 = x.mean(), y.mean()
        sd1, sd2 = x.std(), y.std()
        pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_sd
    logger.info("\n--- t 检验及效应量 (Cohen's d) ---")
    t_stat_100, p_val_100 = stats.ttest_ind(group_100, group_no_limit, equal_var=False)
    d_100 = cohen_d(group_100, group_no_limit)
    t_stat_120, p_val_120 = stats.ttest_ind(group_120, group_no_limit, equal_var=False)
    d_120 = cohen_d(group_120, group_no_limit)
    t_stat_nolimit, p_val_nolimit = stats.ttest_ind(group_100, group_120, equal_var=False)
    d_nolimit = cohen_d(group_100, group_120)

    logger.info("t检验结果：")
    logger.info(f"100 km/h 限速组 vs 无速限组： t统计量 = {t_stat_100:.6f}, p值 = {p_val_100:.6f}, Cohen's d = {d_100:.6f}")
    logger.info(f"120 km/h 限速组 vs 无速限组： t统计量 = {t_stat_120:.6f}, p值 = {p_val_120:.6f}, Cohen's d = {d_120:.6f}")
    logger.info(f"100 km/h 限速组 vs 120km/h 限速组： t统计量 = {t_stat_nolimit:.6f}, p值 = {p_val_nolimit:.6f}, "
                f"Cohen's d = {d_nolimit:.6f}")

    # 单因素方差分析（ANOVA）
    f_stat, p_val_anova = stats.f_oneway(group_100, group_120, group_no_limit)
    logger.info("\nANOVA结果：")
    logger.info(f"F统计量 = {f_stat:.6f}, p值 = {p_val_anova:.6f}")
    # 使用公式 eta² = (F * (k-1)) / (F * (k-1) + (N - k))
    k = 3  # 分组数
    N_total = len(group_100) + len(group_120) + len(group_no_limit)
    df_between = k - 1
    df_within = N_total - k
    eta_squared_anova = (f_stat * df_between) / (f_stat * df_between + df_within)
    logger.info(f"ANOVA效应量 (eta²) = {eta_squared_anova:.6f}")

    # Kruskal-Wallis检验（非参数检验），并计算效应量
    kw_stat, p_val_kw = stats.kruskal(group_100, group_120, group_no_limit)
    logger.info("\nKruskal-Wallis检验结果：")
    logger.info(f"Kruskal-Wallis统计量 = {kw_stat:.6f}, p值 = {p_val_kw:.6f}")

    # 计算效应量，使用公式 eta^2 = (H - k + 1) / (N - k)
    k = 3  # 分组数
    N = len(group_100) + len(group_120) + len(group_no_limit)
    eta_squared = (kw_stat - k + 1) / (N - k)
    logger.info(f"Kruskal-Wallis效应量 (eta^2) = {eta_squared:.6f}")

    # 可视化各组车速分布（箱线图示例）
    df['speed_group'] = df['locationId'].apply(lambda x: '100 km/h' if x in [0, 1] else ('120 km/h' if x == 4 else 'No Limit'))
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='speed_group', y='lonVelocity', data=df, order=['100 km/h', '120 km/h', 'No Limit'])
    plt.title("各组车速分布箱线图")
    plt.xlabel("限速情况")
    plt.ylabel("车速 (km/h)")
    # 保存图片
    if save_path:
        plt.savefig(save_path + '.png', bbox_inches='tight')
    # plt.show()


def plot_lon_velocity_distribution(csv_file, save_path=None):
    """
    读取CSV文件，绘制不同locationId的lonVelocity分布图和核密度拟合分布图，
    图中标注均值和方差。图像排列为两排：第一排3张，第二排4张。

    参数:
        csv_file (str): CSV文件路径。文件中应包含列名
                        ['recordingId', 'locationId', 'trackId', 'lonVelocity']。

    返回:
        matplotlib.figure.Figure: 绘制的图形对象。
    """
    # 读取CSV数据
    df = pd.read_csv(csv_file)
    # 删除 lonVelocity 为负值的记录
    df = df[df['lonVelocity'] > 0]
    # 应用映射函数生成 locationId 列
    df["locationId"] = df["recordingId"].apply(map_recording_to_location)
    # 将 locationId 转为字符串，以便排序和作为分类变量
    df["locationId"] = df["locationId"].astype(str)

    # 获取所有唯一的locationId，并排序（假设共有7个locationId）
    unique_locs = sorted(df['locationId'].unique())
    if len(unique_locs) != 7:
        print(f"警告：locationId的数量为 {len(unique_locs)} ，非预期的7个。请检查数据！")

    # 创建一个两排（第一排3个图，第二排4个图）的图表布局
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    axes = []
    # 第一排：3个图（占用前3列）
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
    # 第二排：4个图（占用4列）
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        axes.append(ax)

    # 对每个locationId绘图
    for idx, loc in enumerate(unique_locs):
        ax = axes[idx]
        # 筛选对应locationId的数据
        data = df[df['locationId'] == loc]

        # 绘制直方图和核密度估计图（归一化为密度）
        sns.histplot(data=data, x='lonVelocity', kde=True, stat="density",
                     ax=ax, color="skyblue", edgecolor="black")

        # 计算均值和方差
        mean_val = data['lonVelocity'].mean()
        var_val = data['lonVelocity'].var()

        # 添加均值和方差的文本注释
        ax.text(0.95, 0.95, f"Mean: {mean_val:.2f}\nVariance: {var_val:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=30,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

        # 设置标题和坐标轴标签的字体大小
        ax.set_title(f"Location {loc}", fontsize=24)
        ax.set_xlabel("lonVelocity", fontsize=18)
        ax.set_ylabel("Density", fontsize=18)

        # 调整坐标轴刻度的字体大小
        ax.tick_params(axis='both', which='major', labelsize=16)

    # 如果数据的locationId少于7个，将隐藏多余的子图
    for j in range(len(unique_locs), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    # 保存图片
    if save_path:
        plt.savefig(save_path + '.png', bbox_inches='tight')
    # plt.show()

    return fig


# 定义转换函数，根据 recordingId 返回对应的 key（即 locationId）
def map_recording_to_location(rec):
    try:
        rec_val = int(rec)
    except ValueError:
        return None
    for key, loc_list in recordingMapToLocation.items():
        if rec_val in loc_list:
            return key
    return None


def plot_stacked_histogram(csv_file, save_path=None):
    # 读取CSV文件
    df = pd.read_csv(csv_file)  # 请替换为你的CSV文件路径

    # 根据 recordingId 转换为 locationId
    # 应用映射函数生成 locationId 列
    df["locationId"] = df["recordingId"].apply(map_recording_to_location)

    # 将 locationId 转为字符串，以便排序和作为分类变量
    df["locationId"] = df["locationId"].astype(str)
    location_order = sorted(df["locationId"].unique(), key=lambda x: int(x))

    # 计算每个 locationId 下不同 class 的数量
    class_counts = df.groupby(["locationId", "class"]).size().unstack(fill_value=0)

    # 生成 tab10 调色板，颜色数目为类别数目
    colors = sns.color_palette("tab10", len(class_counts.columns))

    # 创建两个子图：第一个为数量堆积图，第二个为占比堆积图
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # 绘制绝对数量堆积直方图
    class_counts.loc[location_order].plot(kind="bar", stacked=True, color=colors, ax=axes[0])
    axes[0].set_ylabel("Count")
    axes[0].set_title("Stacked Bar Chart of Class Counts by Location ID")
    axes[0].legend(title="Class")
    axes[0].tick_params(axis='x', rotation=0)

    # 计算各 locationId 的占比（各类别数量除以该 locationId 总数）
    class_counts_prop = class_counts.loc[location_order].div(class_counts.loc[location_order].sum(axis=1), axis=0)

    # 绘制占比堆积直方图
    class_counts_prop.plot(kind="bar", stacked=True, color=colors, ax=axes[1])
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("Stacked Bar Chart of Class Proportions by Location ID")
    axes[1].legend(title="Class")
    axes[1].tick_params(axis='x', rotation=0)

    plt.xlabel("Location ID")
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path + '.png', bbox_inches='tight')
        class_counts.to_csv(save_path + '.csv')

    # plt.show()


def analyze_location_records(csv_path, output_prefix="analysis"):
    """
    增强版分析函数（处理数值型小时字段）

    参数:
    csv_path: 合并后的CSV路径
    output_prefix: 分析结果文件前缀

    返回:
    dict: 包含分析结果和文件路径的字典
    """
    try:
        # 优化内存使用的数据类型
        dtype = {
            'locationId': 'category',
            'duration': 'float32',
            'numTracks': 'int32',
            'startTime': 'int8'  # 显式指定为整数
        }

        # 读取数据并验证字段
        df = pd.read_csv(csv_path, dtype=dtype)
        required_cols = ['locationId', 'startTime', 'duration', 'numTracks']
        # logger.debug(df)
        if not set(required_cols).issubset(df.columns):
            missing = set(required_cols) - set(df.columns)
            return {"error": f"Missing required columns: {missing}"}

        # 数据清洗
        # 过滤无效小时值（0-23）
        valid_hours = df['startTime'].between(0, 23, inclusive='both')
        if not valid_hours.all():
            invalid_count = len(df) - valid_hours.sum()
            df = df[valid_hours]
            logger.info(f"过滤掉{invalid_count}条无效小时记录")

        # 核心统计逻辑
        # 按小时统计
        hourly_stats = df.groupby(['locationId', 'startTime'], observed=True).agg(
            total_duration=('duration', 'sum'),
            record_count=('duration', 'count'),
            total_num=('numTracks', 'sum')
        ).reset_index().rename(columns={'startTime': 'hour'})

        # 总时长统计
        total_stats = df.groupby('locationId', observed=True).agg(
            total_duration=('duration', 'sum'),
            avg_duration=('duration', 'mean'),
            total_num=('numTracks', 'sum'),
            avg_num=('numTracks', 'mean'),
            record_count=('duration', 'count'),
            min_hour=('startTime', 'min'),
            max_hour=('startTime', 'max')
        ).reset_index()

        # 将 total_duration 和 avg_duration 换算为小时，并保留一位小数（除以60）
        total_stats['total_duration'] = (total_stats['total_duration'] / 60).round(1)
        total_stats['avg_duration'] = (total_stats['avg_duration'] / 60).round(1)

        # 输出结果
        hourly_file = f"{output_prefix}_hourly.csv"
        total_file = f"{output_prefix}_total.csv"

        hourly_stats.to_csv(hourly_file, index=False)
        total_stats.to_csv(total_file, index=False)

        # 调用 hourly_stats 可视化函数
        viz_hourly = visualize_hourly_stats(hourly_stats, output_prefix=f"{output_prefix}_hourly_stats.png")

        return {
            "hourly_stats_path": hourly_file,
            "total_stats_path": total_file,
            "unique_locations": df['locationId'].nunique(),
            "total_records": len(df),
            "viz_hourly_stat": viz_hourly,
            "sample_stats": {
                "hourly": hourly_stats.head(3).to_dict('records'),
                "total": total_stats.head(3).to_dict('records')
            }
        }

    except Exception as e:
        return {"error": f"分析失败: {str(e)}"}


def merge_csv_files(input_folder, output_file, suffix=".csv", select_columns=None):
    """
    合并指定文件夹中特定后缀的CSV文件，可选择只合并指定列的数据

    参数:
    input_folder (str): 包含CSV文件的输入文件夹路径
    output_file (str): 合并后的输出文件路径
    suffix (str): 需要合并的文件后缀，默认为 ".csv"
    select_columns (list 或 None): 若不为 None，则只读取指定列，加快处理速度并降低内存使用

    返回:
    str: 合并结果信息
    """
    try:
        # 构建文件匹配模式（不区分大小写）
        pattern = os.path.join(input_folder, f"*{suffix}")
        file_list = glob.glob(pattern, recursive=False)

        # 过滤非CSV文件和隐藏文件
        csv_files = [f for f in file_list
                     if f.lower().endswith(suffix.lower())
                     and not os.path.basename(f).startswith(('.', '~'))]

        if not csv_files:
            return "错误：没有找到符合条件的CSV文件"

        # 初始化合并容器
        merged_df = pd.DataFrame()

        # 记录处理进度
        processed_files = 0
        total_files = len(csv_files)

        for i, file_path in enumerate(sorted(csv_files)):
            try:
                # 若指定了 select_columns，则仅读取所需的列
                if select_columns:
                    df = pd.read_csv(file_path, engine='python', encoding_errors='replace', usecols=select_columns)
                else:
                    df = pd.read_csv(file_path, engine='python', encoding_errors='replace')

                # 添加来源文件名列
                df['source_file'] = os.path.basename(file_path)

                # 合并数据（跳过后续文件的header）
                merged_df = pd.concat([merged_df, df], ignore_index=True)

                processed_files += 1
                logger.info(f"已处理 {i + 1}/{total_files}: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"警告：跳过无法解析的文件 {file_path} - {str(e)}")

        if merged_df.empty:
            return "错误：所有文件均为空或无法解析"

        # 输出合并结果
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        # 生成统计信息
        stats = {
            "input_folder": os.path.abspath(input_folder),
            "output_file": os.path.abspath(output_file),
            "total_files": total_files,
            "processed_files": processed_files,
            "merged_rows": len(merged_df),
            "merged_columns": list(merged_df.columns)
        }

        return f"合并成功！\n统计信息：{stats}"

    except Exception as e:
        return f"发生严重错误：{str(e)}"


def visualize_duration_distribution(csv_path, output_image_path="duration_histogram.png"):
    """
    可视化每个 recordingId 的 duration，并用不同颜色区分不同的 locationId。
    横轴为 recordingId（每隔5个单位显示一次刻度），纵轴为 duration，直方图条宽较大、间隙较小，
    并将图例字体调大。

    参数:
    csv_path: 包含数据的CSV文件路径，要求文件中包含 "recordingId", "duration", "locationId" 三列
    output_image_path: 图片保存路径

    返回:
    dict: 包含生成图片路径或错误信息
    """
    try:
        df = pd.read_csv(csv_path)
        # 检查必要的字段是否存在
        required_cols = {"recordingId", "duration", "locationId", "numTracks"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.error(f"缺少必要的列: {missing}")
            return {"error": f"缺少必要的列: {missing}"}

        # 确保 recordingId 为数值类型以便于横轴刻度处理
        df["recordingId"] = pd.to_numeric(df["recordingId"], errors="coerce")
        df = df.dropna(subset=["recordingId"])
        df["recordingId"] = df["recordingId"].astype(int)

        # 按 recordingId 排序，保证绘图顺序
        df.sort_values("recordingId", inplace=True)

        # 为不同的 locationId 指定颜色（直方图部分使用）
        unique_locations = df["locationId"].unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_locations))
        color_map = {loc: palette[i] for i, loc in enumerate(unique_locations)}

        # 创建图形及左侧纵轴（直方图）
        fig, ax1 = plt.subplots(figsize=(12, 8))
        bar_width = 0.9
        bars = ax1.bar(df["recordingId"], df["duration"], width=bar_width,
                       color=df["locationId"].map(color_map))
        ax1.set_xlabel("Recording ID")
        ax1.set_ylabel("Duration")

        # 横轴刻度以5为间隔显示
        min_rec = df["recordingId"].min()
        max_rec = df["recordingId"].max()
        ax1.set_xticks(range(min_rec, max_rec + 1, 5))

        # 在左侧纵轴 y=100 处添加水平黑色虚线，线宽较粗
        ax1.axhline(y=150, color='black', linestyle='--', linewidth=3)

        # 创建右侧纵轴（折线图），统一使用黑色绘制 numTracks 数据
        ax2 = ax1.twinx()
        ax2.plot(df["recordingId"], df["numTracks"], color='black', marker="o", linestyle='-', label="numTracks")
        ax2.set_ylabel("numTracks")

        # 添加标题
        plt.title("Recording Duration and numTracks by Recording ID and Location")

        # 构造图例：
        # 直方图：每个 locationId 使用不同颜色
        handles_bars = [plt.Rectangle((0, 0), 1, 1, color=color_map[loc]) for loc in unique_locations]
        labels_bars = [f"Location: {loc}" for loc in unique_locations]
        # 折线图：统一使用黑色
        line_handle = plt.Line2D([], [], color='black', marker='o', linestyle='-', label='numTracks')

        # 合并图例句柄并添加到左侧坐标轴
        handles = handles_bars + [line_handle]
        labels = labels_bars + ['numTracks']
        ax1.legend(handles, labels, title="Legend", fontsize=12, title_fontsize=14, loc="upper left")

        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()
        return {"histogram_image": output_image_path}
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        return {"error": f"可视化失败: {str(e)}"}


def visualize_hourly_stats(hourly_stats, output_prefix="hourly_stats"):
    """
    生成两张 **堆叠柱状图**：
    1. 第一张：total_duration（不同 locationId 使用高对比度颜色）
    2. 第二张：total_numTracks（不同 locationId 使用高对比度颜色）

    横轴：0-23 小时，纵轴分别为 duration 和 numTracks，使用 **堆叠** 方式累加不同 locationId 的值。
    """

    try:
        # 透视表（按小时聚合，不同 locationId 的值堆叠）
        duration_pivot = hourly_stats.pivot_table(index="hour", columns="locationId", values="total_duration",
                                                  aggfunc="sum").fillna(0)
        numTracks_pivot = hourly_stats.pivot_table(index="hour", columns="locationId", values="total_num",
                                                   aggfunc="sum").fillna(0)

        # 获取 locationId 并分配颜色（使用高对比度调色板）
        locations = duration_pivot.columns
        num_locations = len(locations)
        high_contrast_palette = sns.color_palette("tab10", n_colors=num_locations)  # 选用对比度高的 tab10 调色板

        # 创建画布
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # 绘制 total_duration 的堆叠直方图
        duration_pivot.plot(kind="bar", stacked=True, ax=axes[0], color=high_contrast_palette, alpha=0.9)
        axes[0].set_ylabel("Total Duration")
        axes[0].set_title("Hourly Total Duration (Stacked by Location)")
        axes[0].legend(title="Location ID", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # 绘制 total_numTracks 的堆叠直方图
        numTracks_pivot.plot(kind="bar", stacked=True, ax=axes[1], color=high_contrast_palette, alpha=0.9)
        axes[1].set_xlabel("Hour (0-23)")
        axes[1].set_ylabel("Total numTracks")
        axes[1].set_title("Hourly numTracks (Stacked by Location)")
        axes[1].legend(title="Location ID", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.6)

        plt.xticks(range(0, 24))  # 显示 0-23 小时
        plt.tight_layout()

        # 保存图片
        duration_img = f"{output_prefix}_duration_stacked.png"
        numtracks_img = f"{output_prefix}_numtracks_stacked.png"
        fig.savefig(f"{output_prefix}_combined_stacked.png")
        plt.close(fig)

        return {
            "duration_chart": duration_img,
            "numTracks_chart": numtracks_img,
            "combined_chart": f"{output_prefix}_combined_stacked.png"
        }
    except Exception as e:
        return {"error": f"Hourly stats visualization failed: {str(e)}"}


# 使用示例
if __name__ == "__main__":
    # 配置参数
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"  # 合并后的输出文件
    dataPath = rootPath + "/drone-dataset-tools-master/data/"  # CSV文件所在文件夹
    recording_meta_suffix = "_recordingMeta.csv"  # 需要合并的文件后缀
    tracks_meta_suffix = "_tracksMeta.csv"
    tracks_suffix = "_tracks.csv"
    recording_meta_outputPath = assetPath + "sum_recordingMeta.csv"
    tracks_meta_outputPath = assetPath + "sum_tracksMeta.csv"
    tracks_outputPath = assetPath + "sum_tracks.csv"

    # 执行合并
    recording_meta_result = merge_and_analyze_1(dataPath, recording_meta_outputPath, recording_meta_suffix)
    logger.info(recording_meta_result)
    tracks_meta_result = merge_and_analyze_2(dataPath, tracks_meta_outputPath, tracks_meta_suffix)
    logger.info(tracks_meta_result)
    tracks_result = merge_and_analyze_3(dataPath, tracks_outputPath, tracks_suffix)
    logger.info(tracks_result)
