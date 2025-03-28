# -*- coding = utf-8 -*-
# @Time : 2025/3/26 18:11
# @Author : 王砚轩
# @File : CVAE_2.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import os
import numpy as np
from src.GAN.CGAN import TimeSeriesDataset
from src.figure.TTC_acc_figure import create_output_folder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TransformerEncoderCVAE(nn.Module):
    """
    Transformer Encoder 模块用于编码输入序列和条件信息，
    将时序特征映射到隐变量空间（生成均值和对数方差）。
    """
    def __init__(self, input_dim, condition_dim, seq_len, d_model, num_heads, num_layers, latent_dim):
        super(TransformerEncoderCVAE, self).__init__()
        self.seq_len = seq_len
        # 将输入数据和条件信息拼接后嵌入到 d_model 维度
        self.fc_in = nn.Linear(input_dim + condition_dim, d_model)
        self.relu = nn.ReLU()
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 将整个序列展平后映射到 latent 空间（生成均值和对数方差）
        self.fc_mean = nn.Linear(seq_len * d_model, latent_dim)
        self.fc_logvar = nn.Linear(seq_len * d_model, latent_dim)

    def forward(self, x, condition):
        # x: [batch_size, seq_len, input_dim]
        # condition: [batch_size, seq_len, condition_dim]
        x_cond = torch.cat((x, condition), dim=-1)  # [batch_size, seq_len, input_dim+condition_dim]
        h = self.relu(self.fc_in(x_cond))            # [batch_size, seq_len, d_model]
        h = self.transformer_encoder(h)               # [batch_size, seq_len, d_model]
        h_flat = h.flatten(start_dim=1)               # [batch_size, seq_len*d_model]
        mean = self.fc_mean(h_flat)                   # [batch_size, latent_dim]
        logvar = self.fc_logvar(h_flat)               # [batch_size, latent_dim]
        return mean, logvar


class TransformerDecoderCVAE(nn.Module):
    """
    Transformer Decoder 模块用于将隐变量扩展成整个时序序列，
    并结合条件信息输出重构序列。
    """
    def __init__(self, latent_dim, condition_dim, seq_len, d_model, num_heads, num_layers, output_dim):
        super(TransformerDecoderCVAE, self).__init__()
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        # 将隐变量扩展到整个序列的表示
        self.fc_latent = nn.Linear(latent_dim, seq_len * d_model)
        # 对条件信息做一次线性映射
        self.fc_condition = nn.Linear(condition_dim, d_model)
        # Transformer 解码器也采用 EncoderLayer（这里结构上与编码器类似）
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        # 将 Transformer 输出映射回原始输出维度
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, z, condition):
        # z: [batch_size, latent_dim]
        # condition: [batch_size, seq_len, condition_dim]
        batch_size = z.size(0)
        latent_seq = self.fc_latent(z)            # [batch_size, seq_len*d_model]
        latent_seq = latent_seq.view(batch_size, self.seq_len, -1)  # [batch_size, seq_len, d_model]
        # 将条件信息映射到 d_model 维度（对每个时间步独立映射）
        cond_emb = self.relu(self.fc_condition(condition))  # [batch_size, seq_len, d_model]
        # 简单将两者相加融合
        x_in = latent_seq + cond_emb
        h = self.transformer_decoder(x_in)        # [batch_size, seq_len, d_model]
        out = self.fc_out(h)                      # [batch_size, seq_len, output_dim]
        return out


class CVAE(nn.Module):
    """
    条件变分自编码器（CVAE），采用 Transformer 结构构造编码器与解码器。
    """
    def __init__(self, input_dim, condition_dim, seq_len, d_model, num_heads, num_layers, latent_dim, output_dim):
        super(CVAE, self).__init__()
        self.encoder = TransformerEncoderCVAE(input_dim, condition_dim, seq_len, d_model, num_heads, num_layers, latent_dim)
        self.decoder = TransformerDecoderCVAE(latent_dim, condition_dim, seq_len, d_model, num_heads, num_layers, output_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, condition):
        mean, logvar = self.encoder(x, condition)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z, condition)
        return x_recon, mean, logvar


def loss_function(recon_x, x, mean, logvar):
    """
    计算重建误差（均方误差）和 KL 散度，并将二者相加作为最终损失。
    """
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_loss


if __name__ == '__main__':
    logger.info("Starting CVAE training...")
    # 参数定义
    seq_len = 421
    input_dim = 18         # 例如：6个特征×3辆车
    condition_dim = 1      # 条件信息维度，例如车辆类别、车道等
    output_dim = 18        # 输出的轨迹维度（可与输入相同）
    d_model = 64           # Transformer 隐藏维度
    num_heads = 8          # 注意力头数
    num_layers = 2         # Transformer 层数
    latent_dim = 16        # 隐变量维度
    batch_size = 64
    epochs = 100

    # 数据准备
    logger.info(f"Inputing data...")
    assetPath = os.path.abspath('../../') + '/asset/'
    create_output_folder(assetPath, 'GENERATED_DATA_VAE')

    # 定义三个文件夹路径（本车、前车、后车）
    folder_path_ego = assetPath + "/normalized_data/"  # 本车数据路径
    folder_path_lead = assetPath + "/normalization_surrounding/lead/"  # 前车数据路径
    folder_path_rear = assetPath + "/normalization_surrounding/rear/"  # 后车数据路径
    target_columns = ['lonLaneletPos', 'latLaneCenterOffset',
                      'heading',
                      'lonVelocity',
                      'lonAcceleration', 'latAcceleration'
                      ]
    result_columns = ['ego_lonLaneletPos', 'ego_latLaneCenterOffset',
                      'ego_heading',
                      'ego_lonVelocity',
                      'ego_lonAcceleration', 'ego_latAcceleration', 'lead_lonLaneletPos', 'lead_latLaneCenterOffset',
                      'lead_heading',
                      'lead_lonVelocity', 'lead_lonAcceleration', 'lead_latAcceleration',
                      'rear_lonLaneletPos', 'rear_latLaneCenterOffset',
                      'rear_heading',
                      'rear_lonVelocity', 'rear_lonAcceleration', 'rear_latAcceleration'
                      ]
    condition_column = 'MergingType'
    dataset = TimeSeriesDataset(
        folder_path_ego,
        folder_path_lead,
        folder_path_rear,
        seq_len,
        output_dim,
        target_columns,
        condition_column
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    logger.info(f"Initialize the generator and discriminator.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CVAE(input_dim, condition_dim, seq_len, d_model, num_heads, num_layers, latent_dim, output_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, cond in dataloader:
            x, cond = x.to(device), cond.to(device)
            optimizer.zero_grad()
            x_recon, mean, logvar = model(x, cond)
            loss = loss_function(x_recon, x, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    logger.info("Training finished.")