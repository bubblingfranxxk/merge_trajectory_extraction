# -*- coding = utf-8 -*-
# @Time : 2024/9/10 14:49
# @Author : 王砚轩
# @File : CGAN.py
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
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
target_length = 543
num_bins = 50


class TimeSeriesDataset(Dataset):
    def __init__(self, folder_path_ego, folder_path_lead, folder_path_rear, seq_len, output_dim, target_columns,
                 condition_column):
        self.data, self.condition, self.extra_info = load_csv_files_to_tensor(
            folder_path_ego, folder_path_lead, folder_path_rear, seq_len, output_dim, target_columns, condition_column
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.condition[idx], self.extra_info[idx]


class TransformerGenerator(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, num_heads, num_layers, output_dim, dropout_prob=0.1):
        super(TransformerGenerator, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_prob)

        # Transformer 编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # 线性层
        # logger.debug(input_dim+condition_dim)
        self.fc_in = nn.Linear(input_dim + condition_dim, d_model)  # 注意修改这里的输入维度
        self.fc_out = nn.Linear(d_model, output_dim)

        # 添加 LayerNorm
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.layer_norm_out = nn.LayerNorm(output_dim)

    def forward(self, noise, condition):
        # 结合条件信息
        input_data = torch.cat((noise, condition), dim=2)  # [batch_size, seq_len, input_dim + condition_dim]
        # print(f"input_data shape: {input_data.shape}")  # 检查输入维度

        # 输入数据经过线性层嵌入 d_model 维度
        embedded_data = self.fc_in(input_data)  # [batch_size, seq_len, d_model]
        # print(f"embedded_data shape after fc_in: {embedded_data.shape}")  # 检查嵌入后的维度

        # 应用 LayerNorm（对每个时间步进行归一化）
        embedded_data = self.layer_norm_in(embedded_data)  # [batch_size, seq_len, d_model]

        # Transformer 编码器需要 [seq_len, batch_size, d_model] 格式
        transformer_output = self.transformer_encoder(embedded_data.permute(1, 0, 2))  # [seq_len, batch_size, d_model]

        # 还原维度 [batch_size, seq_len, output_dim]
        output = self.fc_out(transformer_output.permute(1, 0, 2))  # [batch_size, seq_len, output_dim]

        # 最后一层 LayerNorm
        output = self.layer_norm_out(output)  # [batch_size, seq_len, output_dim]

        return output


# 定义判别器：用于区分真实的时间序列和生成的时间序列
class Discriminator(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, num_layers, dropout_prob=0.1):
        super(Discriminator, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 双向 LSTM 乘 2
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, condition):
        # 结合条件信息
        input_data = torch.cat((x, condition), dim=2)  # [batch_size, seq_len, input_dim + condition_dim]

        # LSTM 编码
        rnn_out, _ = self.rnn(input_data)  # [batch_size, seq_len, hidden_dim * 2]

        # 全连接层输出
        out = self.fc(rnn_out[:, -1, :])  # 取最后一个时间步的输出
        return torch.sigmoid(out)


# CGAN 模型：生成器和判别器
class CGAN:
    def __init__(self, generator, discriminator, data_columns, feature_columns, result_columns,gen_lr=5e-3, disc_lr=1e-3):
        self.rootPath = os.path.abspath('../../')
        self.assetPath = self.rootPath + "/asset/"
        self.generativePath = self.assetPath + "/CGAN/"
        self.data_columns = data_columns
        self.feature_columns = feature_columns
        self.result_columns = result_columns
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.epoch = 0
        self.interval = 10

        # 优化器和损失函数
        self.optim_G = optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=disc_lr)
        self.criterion = nn.BCEWithLogitsLoss()

        # 添加学习率调度器：每100个epoch，学习率降低一半
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optim_G, step_size=100, gamma=0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optim_D, step_size=50, gamma=0.5)

    def save_losses_to_csv(self, epoch, d_loss, g_loss, csv_file="losses.csv"):
        """保存损失值到 CSV 文件，使用 pandas 实现"""
        # 新增一行数据
        new_row = pd.DataFrame({
            'Epoch': [epoch],
            'D_LOSS': [d_loss],
            'G_LOSS': [g_loss]
        })

        # 如果文件已存在，则读取、追加后保存；否则直接保存新行
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(csv_file, index=False, encoding='utf-8')

    def save_fake_data_to_csv(self, fake_data, condition, extra_info, epoch, feature_columns, output_folder="output"):
        """保存生成数据和条件到 CSV 文件"""
        os.makedirs(output_folder, exist_ok=True)
        fake_data_file = os.path.join(output_folder, f"fake_data_epoch_{epoch}.csv")
        condition_file = os.path.join(output_folder, f"condition_epoch_{epoch}.csv")

        # 保存 fake_data
        # logger.debug(f"fake data shape:{fake_data.detach().shape}")
        fake_df = pd.DataFrame(fake_data.detach().cpu().numpy().reshape(-1, fake_data.shape[-1]),
                               columns=self.result_columns)
        # 将 extra_info 添加到 DataFrame 中，extra_info 为 [batch_size, seq_len, 2]
        extra_info_np = extra_info.cpu().numpy().reshape(-1, extra_info.shape[-1])
        fake_df["recordingId"] = extra_info_np[:, 0]
        fake_df["trackId"] = extra_info_np[:, 1]
        fake_df.to_csv(fake_data_file, index=False)

        # 保存 condition
        condition_np = condition.cpu().numpy().reshape(-1, condition.shape[-1])  # [N, 8]
        # 定义 MergingType 的类别顺序（与数据加载时的映射一致）
        merging_categories = ['A-B', 'C', 'D', 'E', 'F']  # 对应 one-hot 编码的 0~7 列
        condition_columns = [f'MergingType_{cat}' for cat in merging_categories]

        condition_df = pd.DataFrame(condition_np, columns=condition_columns)
        condition_df.to_csv(condition_file, index=False)

    def plot_feature_distributions(self, fake_data, feature_columns, output_folder="plots"):
        """绘制特征分布直方图和拟合曲线"""
        os.makedirs(output_folder, exist_ok=True)
        fake_data_np = fake_data.detach().cpu().numpy()

        for i, feature in enumerate(feature_columns):
            plt.figure()
            data = fake_data_np[:, :, i].flatten()  # 提取特定特征的数据
            mu, std = norm.fit(data)  # 拟合正态分布
            plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label="Histogram")

            # 绘制拟合曲线
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f"Fit: μ={mu:.2f}, σ={std:.2f}")

            plt.title(f"Feature: {feature}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)

            # 保存图片
            plot_path = os.path.join(output_folder, f"{feature}_distribution_{self.epoch}.png")
            plt.savefig(plot_path)
            plt.close()

    def train(self, dataloader, noise_dim, epochs=1000):
        # 定义 RMSE 惩罚项的权重
        rmse_weight = 1.0

        for epoch in range(epochs):
            all_fake_data = []  # 用于保存所有的 fake_data
            all_conditions = []  # 用于保存所有的 condition
            all_extra_info = []  # 保存 extra_info
            save_available = False
            if epoch % self.interval == 0:
                save_available = True

            for real_data, condition, extra_info in dataloader:
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                condition = condition.to(self.device)
                # logger.debug(f"real_data shape: {real_data.shape}")
                # logger.debug(type(real_data))
                # logger.debug(f"condition shape: {condition.shape}")

                # 生成随机噪声
                # logger.info(f"Noise is generating")
                noise = torch.randn(batch_size, real_data.size(1), noise_dim).to(self.device)
                # logger.debug(f"noise shape: {noise.shape}")
                fake_data = self.generator(real_data, condition)

                # # 打印张量形状用于调试
                # logger.debug(f"real_data shape: {real_data.shape}")
                # logger.debug(f"condition shape: {condition.shape}")
                # logger.debug(f"noise shape: {noise.shape}")
                # logger.debug(f"fake_data shape: {fake_data.shape}")

                # 真实样本的判别损失
                real_pred = self.discriminator(real_data, condition)
                real_loss = self.criterion(real_pred, torch.ones_like(real_pred))

                # 生成样本的判别损失
                fake_pred = self.discriminator(fake_data.detach(), condition)
                fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
                disc_loss = (real_loss + fake_loss) / 2

                # 判别器反向传播
                self.optim_D.zero_grad()
                disc_loss.backward()

                # 判别器梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.optim_D.step()

                # 生成样本的判别损失（欺骗判别器）
                fake_pred = self.discriminator(fake_data, condition)
                gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))

                # # 计算生成器的 JS 散度惩罚项
                # gen_js_divergence = 0
                # # logger.debug(fake_data.size())
                # for i in range(fake_data.size(2)):  # 遍历每个特征列
                #     real_feature = real_data[:, :, i].detach().cpu().numpy().flatten()
                #     fake_feature = fake_data[:, :, i].detach().cpu().numpy().flatten()
                #     temp = JS_div(real_feature, fake_feature, num_bins,
                #                   min(min(real_feature), min(fake_feature)),
                #                   max(max(real_feature), max(fake_feature)))
                #     # logger.info(f"JS div is {temp}.")
                #     # 计算 JS 散度
                #     gen_js_divergence += min(temp, 1.00)

                # 将 JS 散度作为惩罚项加到生成器损失中
                # logger.debug(gen_js_divergence / fake_data.size(2))
                # gen_loss = 0.5 * gen_loss + 0.5 * gen_js_divergence / fake_data.size(2)  # 将 JS 散度加到生成器的损失中

                # 计算 RMSE 惩罚项：生成数据与真实数据之间的均方根误差
                rmse_loss = torch.sqrt(F.mse_loss(fake_data, real_data))
                # 将 RMSE 惩罚项加入生成器损失中
                gen_loss = gen_loss + rmse_weight * rmse_loss

                # 生成器反向传播
                self.optim_G.zero_grad()
                gen_loss.backward()

                # 生成器梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.optim_G.step()

                # 在训练循环中打印梯度范数
                # logger.info("G grad:")
                # for name, param in self.generator.named_parameters():
                #     if param.grad is not None:
                #         logger.debug(f"Generator {name} grad norm: {param.grad.norm().item()}")
                #
                # logger.info("D grad:")
                # for name, param in self.discriminator.named_parameters():
                #     if param.grad is not None:
                #         logger.debug(f"Discriminator {name} grad norm: {param.grad.norm().item()}")

                if save_available:
                    # 保存每个 batch 的 fake_data 和 condition
                    all_fake_data.append(fake_data.detach().cpu())
                    all_conditions.append(condition.detach().cpu())
                    all_extra_info.append(extra_info)  # extra_info 保持原样

            # 每个epoch结束后更新学习率
            self.scheduler_G.step()
            self.scheduler_D.step()

            if epoch % self.interval == 0:
                # 将保存的列表转换为张量
                all_fake_data = torch.cat(all_fake_data, dim=0)  # 拼接所有的 batch
                all_conditions = torch.cat(all_conditions, dim=0)
                all_extra_info = torch.cat(all_extra_info, dim=0)
                self.epoch = epoch
                # 保存损失值到 CSV
                # self.save_losses_to_csv(epoch, disc_loss.item(), gen_loss.item(),
                #                         csv_file=self.assetPath + "losses.csv")

                # 保存生成数据到 CSV
                self.save_fake_data_to_csv(all_fake_data, all_conditions, all_extra_info, epoch, self.data_columns,
                                           output_folder=self.generativePath + "/GENERATED_DATA/")

                # 绘制特征分布
                # self.plot_feature_distributions(all_fake_data, self.feature_columns,
                #                                 output_folder=self.generativePath + "/GENERATED_DATA/")

            logger.info(f"Epoch [{epoch}/{epochs}] | D Loss: {disc_loss.item()} | G Loss: {gen_loss.item()}"
                        # f" | JS Dive: {gen_js_divergence / fake_data.size(2)} "
                        f"| RMSE: {rmse_loss.item()}")


def load_csv_files_to_tensor(folder_path_ego, folder_path_lead, folder_path_rear, seq_len, output_dim, target_columns,
                             condition_column):
    # 用于存储读取后的数据
    data_list = []
    condition_list = []
    extra_info_list = []  # 用于存储 recordingId 和 trackId
    # 定义字符到数字的映射字典
    merging_type_mapping = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4}

    # 遍历自车文件夹中的所有文件
    for ego_file in os.listdir(folder_path_ego):
        if not ego_file.endswith('.csv'):
            continue

        # 解析自车文件名中的场景ID (前两部分)
        parts = ego_file.split('_')
        if len(parts) < 3:
            continue  # 跳过不符合命名规则的文件
        scene_id = f"{parts[0]}_{parts[1]}"  # 例如 "39_65"

        # 构建前车和后车文件名
        lead_file = f"{scene_id}_leadId_trajectory.csv"
        rear_file = f"{scene_id}_rearId_trajectory.csv"

        # 加载自车数据
        ego_path = os.path.join(folder_path_ego, ego_file)
        ego_df = pd.read_csv(ego_path)
        if not set(target_columns).issubset(ego_df.columns):
            raise ValueError(f"自车文件 {ego_file} 缺少目标列 {target_columns}")
        if condition_column not in ego_df.columns:
            raise ValueError(f"自车文件 {ego_file} 缺少条件列 {condition_column}")

        # 提取自车数据和条件
        ego_data = ego_df[target_columns].values

        # 强制自车数据长度为 seq_len
        if ego_data.shape[0] < seq_len:
            # 填充零
            padding = np.zeros((seq_len - ego_data.shape[0], ego_data.shape[1]))
            ego_data = np.vstack([ego_data, padding])
        else:
            # 截断到 seq_len
            ego_data = ego_data[:seq_len]

        # 提取额外信息 recordingId 和 trackId
        if 'recordingId' in ego_df.columns and 'trackId' in ego_df.columns:
            extra_info = ego_df[['recordingId', 'trackId']].values
            if extra_info.shape[0] < seq_len:
                # 用最后一行复制填充
                padding = np.repeat(extra_info[-1].reshape(1, 2), seq_len - extra_info.shape[0], axis=0)
                extra_info = np.vstack([extra_info, padding])
            else:
                extra_info = extra_info[:seq_len]
        else:
            logger.warning(f"recording id or track id not found!!")
        # 保存额外信息
        extra_info_list.append(torch.tensor(extra_info))

        # 加载前车数据（如果存在）
        lead_path = os.path.join(folder_path_lead, lead_file)
        if os.path.exists(lead_path):
            lead_df = pd.read_csv(lead_path)
            lead_data = lead_df[target_columns].values
            # 强制前车数据长度为 seq_len
            if lead_data.shape[0] < seq_len:
                padding = np.zeros((seq_len - lead_data.shape[0], lead_data.shape[1]))
                lead_data = np.vstack([lead_data, padding])
            else:
                lead_data = lead_data[:seq_len]
        else:
            # 生成全零数据（长度直接为 seq_len）
            lead_data = np.zeros((seq_len, len(target_columns)))

        # 加载后车数据（如果存在）
        rear_path = os.path.join(folder_path_rear, rear_file)
        if os.path.exists(rear_path):
            rear_df = pd.read_csv(rear_path)
            if not set(target_columns).issubset(rear_df.columns):
                raise ValueError(f"后车文件 {rear_file} 缺少目标列 {target_columns}")
            rear_data = rear_df[target_columns].values
        else:
            # 用零矩阵替代缺失的后车数据
            rear_data = np.zeros((seq_len, len(target_columns)))

        # 合并三类数据（按特征维度）
        combined_data = np.hstack([ego_data, lead_data, rear_data])
        # logger.debug(f"combined data shape:{combined_data.shape}")

        # 填充条件数据到 seq_len（用最后一个值填充）
        condition_data = ego_df[condition_column].map(merging_type_mapping).values
        if len(condition_data) < seq_len:
            condition_data = np.concatenate([
                condition_data,
                np.full(seq_len - len(condition_data), condition_data[-1])
            ])
        else:
            condition_data = condition_data[:seq_len]
        # logger.debug(f"conditional data shape{condition_data.shape}")

        # 添加到列表
        data_list.append(torch.tensor(combined_data, dtype=torch.float32))
        condition_list.append(torch.tensor(condition_data, dtype=torch.float32))

    # 合并所有场景数据
    real_data = torch.stack(data_list, dim=0)  # [batch_size, seq_len, output_dim]
    condition = torch.stack(condition_list, dim=0).unsqueeze(-1)  # [batch_size, seq_len, 1]

    # 获取类别数 (假设 condition 中的值是类别索引，且类别数已知)
    num_classes = 5  # 假设有8个类别

    # 使用 one_hot 函数转换为 one-hot 编码
    # 通过 squeeze 去掉最后一维，以便使用 one_hot 函数
    condition_one_hot = F.one_hot(condition.long().squeeze(-1), num_classes=num_classes)
    # 将额外信息堆叠起来
    extra_info = torch.stack(extra_info_list, dim=0)  # [batch_size, seq_len, 2]
    # 保存条件数据到 CSV（调试用）
    _save_condition_to_csv(condition_one_hot, "condition_debug.csv")

    return real_data, condition_one_hot, extra_info
    # 遍历文件夹中的所有CSV文件


def _save_condition_to_csv(condition_tensor, filename):
    """将条件张量保存为 CSV 文件"""
    # 转换为 numpy 数组并调整形状
    condition_np = condition_tensor.cpu().numpy().reshape(-1, condition_tensor.shape[-1])  # [batch_size * seq_len, 8]

    # 定义列名（按 MergingType 字母顺序）
    merging_categories = ['A-B', 'C', 'D', 'E', 'F']  # 与 merging_type_mapping 顺序一致
    condition_columns = [f'MergingType_{cat}' for cat in merging_categories]

    # 创建 DataFrame 并保存
    condition_df = pd.DataFrame(condition_np, columns=condition_columns)
    condition_df.to_csv(assetPath + filename, index=False)
    print(f"条件数据已保存至 {filename}")


if __name__ == '__main__':
    logger.warning(f"cuda available: {torch.cuda.is_available()}.")

    # 参数定义
    input_dim = 6 * 3  # 输入时间序列的维度
    seq_len = target_length  # 时间序列长度
    d_model = 512  # Transformer 的隐藏维度
    num_heads = 32  # 多头注意力头的数量
    num_layers = 4  # Transformer 编码器层数
    hidden_dim = 128  # 判别器的 LSTM 隐藏层维度
    noise_dim = 6 * 3  # 生成器输入噪声的维度
    output_dim = 6 * 3  # 生成的时间序列维度
    condition_dim = 5  # 条件维度，例如特征、标签等

    batch_size = 64

    # 数据准备
    logger.info(f"Inputing data...")
    assetPath = os.path.abspath('../../') + '/asset/'
    CGAN_path = assetPath + '/CGAN/'
    create_output_folder(CGAN_path, 'GENERATED_DATA')
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
    # , 'RearTTCRaw3', 'LeadTTCRaw3',
    # 'LeftRearTTCRaw3', 'LeftLeadTTCRaw3', 'LeftAlongsideTTCRaw3']
    features = ['lonVelocity', 'lonAcceleration', 'latAcceleration']
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

    # 初始化生成器和判别器
    logger.info(f"Initialize the generator and discriminator.")
    generator = TransformerGenerator(input_dim, seq_len, d_model, num_heads, num_layers, output_dim)
    discriminator = Discriminator(output_dim + condition_dim, seq_len, hidden_dim, num_layers)

    # 初始化 CGAN
    logger.info(f"Initialize CGAN model.")
    cgan = CGAN(generator, discriminator, target_columns, features, result_columns)

    # 生成随机时间序列数据和条件
    # real_data = torch.randn(32, seq_len, output_dim)  # [batch_size, seq_len, output_dim]
    # condition = torch.randn(32, seq_len, condition_dim)  # [batch_size, seq_len, condition_dim]
    # logger.info(real_data)
    # logger.warning(condition)
    # logger.info(real_data.shape)

    # 训练 CGAN
    logger.info(f"Model is training...")
    cgan.train(dataloader, noise_dim, epochs=500)
