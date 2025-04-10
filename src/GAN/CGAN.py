# -*- coding = utf-8 -*-
# @Time : 2024/9/10 14:49
# @Author : 王砚轩
# @File : GAN.py
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
target_length = 543
num_bins = 50


class TimeSeriesDataset(Dataset):
    def __init__(self, folder_path_ego, folder_path_lead, folder_path_rear, seq_len, output_dim, target_columns,
                 condition_column):
        self.data, self.condition, self.extra_info, self.mask = load_csv_files_to_tensor(
            folder_path_ego,
            folder_path_lead,
            folder_path_rear,
            seq_len,
            output_dim,
            target_columns,
            condition_column
        )

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.condition[idx], self.extra_info[idx], self.mask[idx]


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

    def forward(self, noise, condition, mask):
        # mask: [B, seq_len, 3] —— True 表示“有效”，0 表示 pad
        # 先算出 src_key_padding_mask: [B, seq_len], True=要屏蔽
        # 如果任一车种缺失，就屏蔽该时间步
        step_mask = (mask.min(dim=2).values == 0)  # [B, seq_len], bool

        x = torch.cat((noise, condition), dim=2)  # [B, seq_len, input_dim+cond_dim]
        x = self.fc_in(x)
        x = self.layer_norm_in(x)

        # TransformerEncoder 支持 src_key_padding_mask
        # 注意传入的 mask: True 表示“不要 attend”
        transformer_out = self.transformer_encoder(
            x, src_key_padding_mask=step_mask
        )  # [B, seq_len, d_model]

        out = self.fc_out(transformer_out)
        out = self.layer_norm_out(out)
        return out


# 定义判别器：用于区分真实的时间序列和生成的时间序列
class Discriminator(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, num_layers, dropout_prob=0.1):
        super(Discriminator, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 双向 LSTM 乘 2
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, condition, mask):
        input_data = torch.cat((x, condition), dim=2)  # [B, seq_len, feat+cond]

        # 计算每条序列的有效长度（在 GPU 上）
        lengths = mask[:, :, 0].sum(dim=1).long()  # [B], cuda LongTensor

        # **移动到 CPU** 并确保 dtype 是 int64
        lengths_cpu = lengths.cpu()

        # pack_padded_sequence 要求 lengths 在 CPU 上
        packed = pack_padded_sequence(input_data,
                                      lengths_cpu,
                                      batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out,
                                         batch_first=True,
                                         total_length=input_data.size(1))

        # 取每条序列最后一个有效输出
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, rnn_out.size(2))  # 这里 lengths 用原始 cuda 张量没问题
        last = rnn_out.gather(1, idx).squeeze(1)

        out = self.fc(last)
        return torch.sigmoid(out)


# GAN 模型：生成器和判别器
class CGAN:
    def __init__(self, generator, discriminator, data_columns, feature_columns, result_columns,gen_lr=1e-3, disc_lr=5e-4):
        self.rootPath = os.path.abspath('../../')
        self.assetPath = self.rootPath + "/asset/"
        self.generativePath = self.assetPath + "/GAN/"
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

    def save_losses_to_csv(self, epoch, d_loss, g_loss, temp_loss, csv_file="losses.csv"):
        """保存损失值到 CSV 文件，使用 pandas 实现"""
        # 新增一行数据
        new_row = pd.DataFrame({
            'Epoch': [epoch],
            'D_LOSS': [d_loss],
            'G_LOSS': [g_loss],
            'T_LOSS': [temp_loss]
        })

        # 如果文件已存在，则读取、追加后保存；否则直接保存新行
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(csv_file, index=False, encoding='utf-8')

    def save_fake_data_to_csv(self,
                              fake_data,  # Tensor [B, seq_len, D]
                              condition,  # Tensor [B, seq_len, C]
                              extra_info,  # Tensor [B, seq_len, 2]
                              mask,  # Tensor [B, seq_len, 3]
                              epoch,
                              feature_columns,
                              output_folder="output"):
        """
        保存生成数据和条件到 CSV 文件，但只保留 ego_mask == 1 的行，
        并在 fake_data CSV 中同时输出 lead_mask 和 rear_mask。
        """
        os.makedirs(output_folder, exist_ok=True)
        fake_data_file = os.path.join(output_folder, f"fake_data_epoch_{epoch}.csv")
        condition_file = os.path.join(output_folder, f"condition_epoch_{epoch}.csv")

        # —— 1. 展平 fake_data、extra_info、mask ——
        B, seq, D = fake_data.shape
        fake_np = fake_data.detach().cpu().numpy().reshape(-1, D)  # [(B*seq), D]
        extra_np = extra_info.cpu().numpy().reshape(-1, 2)  # [(B*seq), 2]
        mask_np = mask.detach().cpu().numpy().reshape(-1, 3)  # [(B*seq), 3]

        # —— 2. 构造 fake_df 并附加 recordingId/trackId/masks ——
        fake_df = pd.DataFrame(fake_np, columns=self.result_columns)
        fake_df["recordingId"] = extra_np[:, 0].astype(int)
        fake_df["trackId"] = extra_np[:, 1].astype(int)
        fake_df["ego_mask"] = mask_np[:, 0].astype(int)
        fake_df["lead_mask"] = mask_np[:, 1].astype(int)
        fake_df["rear_mask"] = mask_np[:, 2].astype(int)

        # —— 3. 只保留 ego_mask == 1 的行 ——
        fake_df = fake_df[fake_df["ego_mask"] == 1].reset_index(drop=True)

        # —— 4. 保存 fake_data CSV ——
        fake_df.to_csv(fake_data_file, index=False, encoding='utf-8')

        # —— 5. 处理 condition ——
        # 展平并构造 DataFrame
        cond_np = condition.cpu().numpy().reshape(-1, condition.shape[-1])  # [(B*seq), C]
        merging_categories = ['A-B', 'C', 'D', 'E', 'F']
        condition_columns = [f"MergingType_{cat}" for cat in merging_categories]
        cond_df = pd.DataFrame(cond_np, columns=condition_columns)
        # 同样附加 ego_mask，以便筛选
        cond_df["ego_mask"] = mask_np[:, 0].astype(int)
        # 只保留 ego_mask == 1
        cond_df = cond_df[cond_df["ego_mask"] == 1].reset_index(drop=True)
        # 不需要把 ego_mask 存到 condition 文件里，删掉
        cond_df = cond_df.drop(columns=["ego_mask"])

        # —— 6. 保存 condition CSV ——
        cond_df.to_csv(condition_file, index=False, encoding='utf-8')

        logger.info(f"Saved filtered fake data to {fake_data_file} and condition to {condition_file}")

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
        rmse_weight = 2.0

        for epoch in range(epochs):
            all_fake_data = []  # 用于保存所有的 fake_data
            all_conditions = []  # 用于保存所有的 condition
            all_extra_info = []  # 保存 extra_info
            save_available = False
            all_masks = []  # 新增：收集 mask
            if epoch % self.interval == 0:
                save_available = True

            for real_data, condition, extra_info, mask in dataloader:
                real_data = real_data.to(self.device)
                condition = condition.to(self.device)
                mask = mask.to(self.device)
                # logger.debug(f"real_data shape: {real_data.shape}")
                # logger.debug(type(real_data))
                # logger.debug(f"condition shape: {condition.shape}")

                # 1) 生成 fake_data
                # logger.info(f"Noise is generating")
                noise = torch.randn(real_data.size(0),
                                    real_data.size(1),
                                    noise_dim,
                                    device=self.device)
                # logger.debug(f"noise shape: {noise.shape}")
                fake_data = self.generator(real_data, condition, mask)

                # # 打印张量形状用于调试
                # logger.debug(f"real_data shape: {real_data.shape}")
                # logger.debug(f"condition shape: {condition.shape}")
                # logger.debug(f"noise shape: {noise.shape}")
                # logger.debug(f"fake_data shape: {fake_data.shape}")

                # 2) 判别器 loss
                # 真实样本的判别损失
                real_pred = self.discriminator(real_data, condition, mask)
                real_loss = self.criterion(real_pred, torch.ones_like(real_pred))

                # 生成样本的判别损失
                fake_pred = self.discriminator(fake_data.detach(), condition, mask)
                fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
                disc_loss = (real_loss + fake_loss) / 2

                # 判别器反向传播
                self.optim_D.zero_grad()
                disc_loss.backward()

                # 判别器梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.optim_D.step()

                # 生成样本的判别损失（欺骗判别器）
                fake_pred_for_g  = self.discriminator(fake_data, condition, mask)
                gen_loss = temp_loss = self.criterion(fake_pred_for_g, torch.ones_like(fake_pred_for_g))

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

                # MSE 部分：只在 mask=1 的位置计算
                mse_all = F.mse_loss(fake_data, real_data, reduction='none')  # [B,seq,feat]
                # 扩展 mask 到 feature 维度：ego/lead/rear 各自重复 target_columns 长度次
                f = len(target_columns)
                mask_feat = torch.cat([
                    mask[:, :, 0:1].expand(-1, -1, f),
                    mask[:, :, 1:2].expand(-1, -1, f),
                    mask[:, :, 2:3].expand(-1, -1, f),
                ], dim=2)  # [B,seq,feat*3]
                weighted_mse = mse_all * mask_feat
                rmse_loss = torch.sqrt(weighted_mse.sum() / mask_feat.sum())

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
                    all_masks.append(mask.detach().cpu())  # 新增

            # 每个epoch结束后更新学习率
            self.scheduler_G.step()
            self.scheduler_D.step()

            if epoch % self.interval == 0:
                # 将保存的列表转换为张量
                all_fake_data = torch.cat(all_fake_data, dim=0)  # 拼接所有的 batch
                all_conditions = torch.cat(all_conditions, dim=0)
                all_extra_info = torch.cat(all_extra_info, dim=0)
                all_masks = torch.cat(all_masks, dim=0)  # [B, seq, 3]
                self.epoch = epoch
                # 保存损失值到 CSV
                self.save_losses_to_csv(epoch, disc_loss.item(), gen_loss.item(), temp_loss.item(),
                                        csv_file=self.assetPath + "losses.csv")

                # 保存生成数据到 CSV
                self.save_fake_data_to_csv(all_fake_data, all_conditions, all_extra_info, all_masks, epoch, self.data_columns,
                                           output_folder=self.generativePath + "/GENERATED_DATA/")

                # 绘制特征分布
                # self.plot_feature_distributions(all_fake_data, self.feature_columns,
                #                                 output_folder=self.generativePath + "/GENERATED_DATA/")

            logger.info(f"Epoch [{epoch}/{epochs}] | D Loss: {disc_loss.item()} | G Loss: {temp_loss.item()}"
                        # f" | JS Dive: {gen_js_divergence / fake_data.size(2)} "
                        f"| RMSE: {rmse_loss.item()}")


def load_csv_files_to_tensor(folder_path_ego, folder_path_lead, folder_path_rear, seq_len, output_dim, target_columns,
                             condition_column):
    """
    读取 ego/lead/rear 三组 CSV，按 ego 的 frame 对齐，
    生成 data、condition、extra_info 以及 mask 张量：
      - data:    [B, seq_len, output_dim]
      - condition: [B, seq_len, num_classes]
      - extra_info: [B, seq_len, 2] （recordingId, trackId）
      - mask:    [B, seq_len, 3] （ego, lead, rear）
    """
    # 合并后 data 的特征维度应该等于 output_dim
    # mask 最后一维为 3，分别对应 ego/lead/rear
    data_list = []
    condition_list = []
    extra_info_list = []
    mask_list = []
    # 映射 MergingType 字符到类别索引
    merging_type_mapping = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4}
    num_classes = 5  # 根据 mapping 中最大索引 + 1

    def pad_seq(arr, mask):
        """
        将 arr: [N, D] 和 mask: [N,1] pad 或截断到 [seq_len, D] / [seq_len,1]
        pad 部分用 0 填充
        """
        L = arr.shape[0]
        if L < seq_len:
            pad_len = seq_len - L
            arr = np.vstack([arr, np.zeros((pad_len, arr.shape[1]))])
            mask = np.vstack([mask, np.zeros((pad_len, 1))])
        else:
            arr = arr[:seq_len]
            mask = mask[:seq_len]
        return arr, mask

    # 遍历自车文件夹中的所有文件
    for ego_file in os.listdir(folder_path_ego):
        if not ego_file.endswith('.csv'):
            continue

        # —— 1. 读取 ego 数据 ——
        ego_path = os.path.join(folder_path_ego, ego_file)
        ego_df = pd.read_csv(ego_path)

        # 用 ego 的 frame 作为公共时间基准
        frames = ego_df['frame'].values  # shape [N]
        N = len(frames)

        # ego_data + ego_mask（全 1）
        ego_data = ego_df[target_columns].values  # [N, feat]
        ego_mask = np.ones((N, 1), dtype=float)  # [N,1]

        # —— 2. 读取 lead ——
        scene_id = '_'.join(ego_file.split('_')[:2])
        lead_file = f"{scene_id}_leadId_trajectory.csv"
        lead_path = os.path.join(folder_path_lead, lead_file)
        # logger.debug(lead_path)
        if os.path.exists(lead_path):
            # l = len(pd.read_csv(lead_path))
            # logger.debug(f"L_{scene_id}: {l}")
            # logger.debug(f"N_{scene_id}: {N}")
            # if l == N:
            #     continue
            lead_df = pd.read_csv(lead_path).set_index('frame').reindex(frames)
            # 如果某行任一 target_columns 为 NaN，则视作缺失
            lead_mask = (~lead_df[target_columns].isna().any(axis=1)).astype(float).values.reshape(N, 1)
            lead_df.fillna(0, inplace=True)
            lead_data = lead_df[target_columns].values
        else:
            lead_data = np.zeros((N, len(target_columns)))
            lead_mask = np.zeros((N, 1), dtype=float)

        # —— 3. 读取 rear ——
        rear_file = f"{scene_id}_rearId_trajectory.csv"
        rear_path = os.path.join(folder_path_rear, rear_file)
        if os.path.exists(rear_path):
            rear_df = pd.read_csv(rear_path).set_index('frame').reindex(frames)
            rear_mask = (~rear_df[target_columns].isna().any(axis=1)).astype(float).values.reshape(N, 1)
            rear_df.fillna(0, inplace=True)
            rear_data = rear_df[target_columns].values
        else:
            rear_data = np.zeros((N, len(target_columns)))
            rear_mask = np.zeros((N, 1), dtype=float)

        # —— 4. pad / truncate 到 seq_len ——
        ego_data, ego_mask = pad_seq(ego_data, ego_mask)
        lead_data, lead_mask = pad_seq(lead_data, lead_mask)
        rear_data, rear_mask = pad_seq(rear_data, rear_mask)

        # —— 5. 合并 data 与 mask ——
        # data: [seq_len, feat*3], mask: [seq_len, 3]
        combined_data = np.hstack([ego_data, lead_data, rear_data])
        combined_mask = np.hstack([ego_mask, lead_mask, rear_mask])

        # —— 6. 构造 condition_one_hot ——
        # 先从 ego_df 取原始条件值，并 pad/truncate
        cond_vals = ego_df[condition_column].map(merging_type_mapping).values  # [N]
        if len(cond_vals) < seq_len:
            cond_vals = np.concatenate([
                cond_vals,
                np.full(seq_len - len(cond_vals), cond_vals[-1], dtype=int)
            ])
        else:
            cond_vals = cond_vals[:seq_len]
        # 转为 tensor 并做 one-hot
        cond_tensor = torch.tensor(cond_vals, dtype=torch.long)  # [seq_len]
        condition_one_hot = F.one_hot(cond_tensor, num_classes=num_classes)  # [seq_len, num_classes]
        condition_one_hot = condition_one_hot.to(torch.float32)

        # —— 7. 构造 extra_info ——
        # 从 ego_df 取 recordingId, trackId
        extra = ego_df[['recordingId', 'trackId']].values  # [N,2]
        if extra.shape[0] < seq_len:
            last = extra[-1].reshape(1, 2)
            extra = np.vstack([extra, np.repeat(last, seq_len - extra.shape[0], axis=0)])
        else:
            extra = extra[:seq_len]
        extra_info = torch.tensor(extra, dtype=torch.float32)  # [seq_len,2]

        # —— 8. 收集到列表 ——
        data_list.append(torch.tensor(combined_data, dtype=torch.float32))  # [seq_len, output_dim]
        mask_list.append(torch.tensor(combined_mask, dtype=torch.float32))  # [seq_len, 3]
        condition_list.append(condition_one_hot)  # [seq_len, num_classes]
        extra_info_list.append(extra_info)  # [seq_len, 2]

    # —— 9. 堆叠成 batch ——
    real_data = torch.stack(data_list, dim=0)  # [B, seq_len, output_dim]
    mask = torch.stack(mask_list, dim=0)  # [B, seq_len, 3]
    condition = torch.stack(condition_list, dim=0)  # [B, seq_len, num_classes]
    extra_info = torch.stack(extra_info_list, dim=0)  # [B, seq_len, 2]

    return real_data, condition, extra_info, mask


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
    num_heads = 64  # 多头注意力头的数量
    num_layers = 4  # Transformer 编码器层数
    hidden_dim = 128  # 判别器的 LSTM 隐藏层维度
    noise_dim = 6 * 3  # 生成器输入噪声的维度
    output_dim = 6 * 3  # 生成的时间序列维度
    condition_dim = 5  # 条件维度，例如特征、标签等

    batch_size = 32

    # 数据准备
    logger.info(f"Inputing data...")
    assetPath = os.path.abspath('../../') + '/asset/'
    CGAN_path = assetPath + '/GAN/'
    create_output_folder(CGAN_path, 'GENERATED_DATA')
    # 定义三个文件夹路径（本车、前车、后车）
    folder_path_ego = assetPath + "/normalized_data/"  # 本车数据路径
    folder_path_lead = assetPath + "/normalization_surrounding/leadId/"  # 前车数据路径
    folder_path_rear = assetPath + "/normalization_surrounding/rearId/"  # 后车数据路径
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

    # 初始化 GAN
    logger.info(f"Initialize GAN model.")
    cgan = CGAN(generator, discriminator, target_columns, features, result_columns)

    # 生成随机时间序列数据和条件
    # real_data = torch.randn(32, seq_len, output_dim)  # [batch_size, seq_len, output_dim]
    # condition = torch.randn(32, seq_len, condition_dim)  # [batch_size, seq_len, condition_dim]
    # logger.info(real_data)
    # logger.warning(condition)
    # logger.info(real_data.shape)

    # 训练 GAN
    logger.info(f"Model is training...")
    cgan.train(dataloader, noise_dim, epochs=500)
