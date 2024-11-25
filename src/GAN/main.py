# -*- coding = utf-8 -*-
# @Time : 2024/9/10 18:56
# @Author : 王砚轩
# @File : main.py
# @Software: PyCharm
import os
from loguru import logger
from CSVDataLoader import CSVDataLoader
from CSVDataLoader import DataLoader
import pandas as pd

# 使用示例
if __name__ == "__main__":
    csv_file = os.path.abspath('../../') + r'\asset\trajMatchMergingType.csv'
    df = pd.read_csv(csv_file)
    feature_cols = df.columns[3:124]
    feature_cols.append(pd.Index([df.columns[139]]))
    logger.info(feature_cols)
    target_col = df.columns[124:139]
    logger.info(target_col)

    data_loader = CSVDataLoader(csv_file, feature_cols, target_col, batch_size=32, shuffle=True)
    dataloader = data_loader.get_dataloader()





