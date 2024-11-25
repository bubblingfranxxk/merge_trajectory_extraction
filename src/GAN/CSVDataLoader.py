# -*- coding = utf-8 -*-
# @Time : 2024/9/10 18:55
# @Author : 王砚轩
# @File : CSVDataLoader.py
# @Software: PyCharm
import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from loguru import logger


class CSVDataLoader:
    def __init__(self, csv_file, feature_cols, target_col, batch_size=32, shuffle=True):
        self.csv_file = csv_file  # 文件路径设置为成员变量
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 读取和预处理数据
        self.features, self.targets = self._load_and_preprocess_data()
        self.dataset = MyDataset(self.features, self.targets)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def _load_and_preprocess_data(self):
        # 读取 CSV 文件
        data = pd.read_csv(self.csv_file)

        # 数据预处理
        data = data.dropna()
        features = data[self.feature_cols].values
        targets = data[self.target_col].values

        # 数据标准化
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # 转换为张量
        features_tensor = torch.tensor(features, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        return features_tensor, targets_tensor

    def get_dataloader(self):
        return self.dataloader

    def traverse_data(self):
        for batch_features, batch_targets in self.dataloader:
            print("Features:", batch_features)
            print("Targets:", batch_targets)


class MyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]



