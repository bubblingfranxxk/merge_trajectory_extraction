# -*- coding = utf-8 -*-
# @Time : 2025/3/26 18:08
# @Author : 王砚轩
# @File : CVAE.py
# @Software: PyCharm

# -*- coding = utf-8 -*-
# @Time : 2024/9/10 14:49
# @Author : 王砚轩
# @File : CVAE.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loguru import logger
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.figure.TTC_acc_figure import create_output_folder
from utils.common import JS_div

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
target_length = 421
num_bins = 50


class TimeSeriesDataset(Dataset):
    # 保持原有数据加载逻辑不变
    def __init__(self, folder_path_ego, folder_path_lead, folder_path_rear, seq_len, output_dim, target_columns,
                 condition_column):
        self.data, self.condition = self.load_csv_files_to_tensor(
            folder_path_ego, folder_path_lead, folder_path_rear, seq_len, output_dim, target_columns, condition_column
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.condition[idx]

    # 保持原有数据加载方法
    def load_csv_files_to_tensor(self, folder_path_ego, folder_path_lead, folder_path_rear, seq_len, output_dim,
                                 target_columns, condition_column):
        # 保持原有实现不变
        # ...
        return real_data, condition


class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, seq_len, hidden_dim=128, num_layers=2):
        super(CVAE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim + condition_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim + condition_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, c):
        # 拼接条件和输入数据
        x = torch.cat([x, c], dim=-1)  # [batch, seq, input_dim + cond_dim]
        _, (h_n, _) = self.encoder_lstm(x)
        hidden = h_n[-1]  # 取最后一层的隐藏状态
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # 扩展条件信息到序列长度
        batch_size = c.size(0)
        c_expanded = c.expand(-1, self.seq_len, -1)

        # 扩展潜在变量到序列长度
        z_expanded = z.unsqueeze(1).expand(-1, self.seq_len, -1)

        # 拼接潜在变量和条件
        decoder_input = torch.cat([z_expanded, c_expanded], dim=-1)

        # LSTM解码
        output, _ = self.decoder_lstm(decoder_input)
        reconstructed = self.decoder_fc(output)
        return reconstructed

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


class CVAETrainer:
    def __init__(self, model, data_columns, feature_columns, result_columns, lr=1e-3):
        self.rootPath = os.path.abspath('../../')
        self.assetPath = self.rootPath + "/asset/"
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.epoch = 0
        self.interval = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 保持原有可视化参数
        self.data_columns = data_columns
        self.feature_columns = feature_columns
        self.result_columns = result_columns

    def cvae_loss(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # 保持原有可视化方法
    def save_losses_to_csv(self, epoch, loss, csv_file="cvae_losses.csv"):
        """保持原有实现不变"""

    def save_fake_data_to_csv(self, fake_data, condition, epoch, output_folder="cvae_output"):
        """保持原有实现不变"""

    def plot_feature_distributions(self, fake_data, output_folder="cvae_plots"):
        """保持原有实现不变"""

    def train(self, dataloader, epochs=500):
        for epoch in range(epochs):
            total_loss = 0
            all_recon = []
            all_conditions = []

            for batch_idx, (real_data, condition) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                condition = condition.to(self.device)

                # 前向传播
                recon_data, mu, logvar = self.model(real_data, condition)

                # 计算损失
                loss = self.cvae_loss(recon_data, real_data, mu, logvar)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # 保存生成结果
                if epoch % self.interval == 0:
                    all_recon.append(recon_data.detach().cpu())
                    all_conditions.append(condition.detach().cpu())

            # 周期日志和可视化
            if epoch % self.interval == 0:
                avg_loss = total_loss / len(dataloader.dataset)
                self.save_losses_to_csv(epoch, avg_loss)

                # 拼接所有生成数据
                all_recon = torch.cat(all_recon, dim=0)
                all_conditions = torch.cat(all_conditions, dim=0)

                # 保存和可视化
                self.save_fake_data_to_csv(all_recon, all_conditions, epoch)
                self.plot_feature_distributions(all_recon)

                logger.info(f"Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    logger.warning(f"CUDA available: {torch.cuda.is_available()}")

    # 参数配置（保持与原有代码一致）
    input_dim = 6 * 3  # 输入特征维度
    seq_len = target_length
    condition_dim = 1  # MergingType条件维度
    latent_dim = 32
    hidden_dim = 128
    num_layers = 2

    # 初始化模型
    cvae = CVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    # 初始化训练器
    trainer = CVAETrainer(
        model=cvae,
        data_columns=target_columns,
        feature_columns=features,
        result_columns=result_columns,
        lr=1e-3
    )

    # 数据加载（保持原有配置）
    dataset = TimeSeriesDataset(
        folder_path_ego,
        folder_path_lead,
        folder_path_rear,
        seq_len,
        output_dim,
        target_columns,
        condition_column
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 开始训练
    logger.info("Start CVAE Training...")
    trainer.train(dataloader, epochs=500)