# -*- coding = utf-8 -*-
# @Time : 2025/4/10 23:20
# @Author : 王砚轩
# @File : ttc_count.py
# @Software: PyCharm
import os
import pandas as pd


def count_files_with_min_ttc(csv_path):
    """
    遍历 csv_path 下所有 CSV 文件，统计满足下列条件的文件数：
      - 文件中存在至少一行数据，其 lead_ttc 和 rear_ttc 两列的最小值小于 3
      - 文件中存在至少一行数据，其 lead_ttc 和 rear_ttc 两列的最小值小于 1.5
      - 文件中存在至少一行数据，其 lead_ttc 和 rear_ttc 两列的最小值小于 1
    每个文件只计 1 或 0，最后返回这三个条件下文件数的计数。

    :param csv_path: 包含 CSV 文件的目录路径
    :return: 一个元组 (count_lt_3, count_lt_1_5, count_lt_1)
    """
    count_lt_3 = 0
    count_lt_1_5 = 0
    count_lt_1 = 0

    # 遍历目录下所有文件
    for filename in os.listdir(csv_path):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(csv_path, filename)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")
                continue

            # 检查必须包含 lead_ttc 和 rear_ttc 两列
            if 'lead_ttc' not in df.columns or 'rear_ttc' not in df.columns:
                print(f"文件 {file_path} 缺少 lead_ttc 或 rear_ttc 列，跳过。")
                continue

            # 对每一行计算两列的最小值
            min_ttc = df[['lead_ttc', 'rear_ttc']].min(axis=1)

            # 如果至少有一行的最小值小于阈值，则计数加 1
            if (min_ttc < 3).any():
                count_lt_3 += 1
            if (min_ttc < 1.5).any():
                count_lt_1_5 += 1
            if (min_ttc < 1).any():
                count_lt_1 += 1

    return count_lt_3, count_lt_1_5, count_lt_1


# 示例调用
if __name__ == "__main__":
    rootPath = os.path.abspath('../../')
    assetPath = os.path.join(rootPath, 'asset')
    csv_directory = os.path.join(assetPath, 'generative_ttc_enCGAN')  # 替换为实际的文件夹路径
    result = count_files_with_min_ttc(csv_directory)
    print("文件中 min(lead_ttc, rear_ttc) < 3 的数量:", result[0])
    print("文件中 min(lead_ttc, rear_ttc) < 1.5 的数量:", result[1])
    print("文件中 min(lead_ttc, rear_ttc) < 1 的数量:", result[2])
