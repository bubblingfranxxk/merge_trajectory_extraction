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


class TransformerGenerator(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, num_heads, num_layers, output_dim):
        super(TransformerGenerator, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model

        # Transformer 编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # 线性层
        self.fc_in = nn.Linear(input_dim + condition_dim, d_model)  # 注意修改这里的输入维度
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, noise, condition):
        # 结合条件信息
        input_data = torch.cat((noise, condition), dim=2)  # [batch_size, seq_len, input_dim + condition_dim]
        # print(f"input_data shape: {input_data.shape}")  # 检查输入维度

        # 输入数据经过线性层嵌入 d_model 维度
        embedded_data = self.fc_in(input_data)  # [batch_size, seq_len, d_model]
        # print(f"embedded_data shape after fc_in: {embedded_data.shape}")  # 检查嵌入后的维度

        # Transformer 编码器需要 [seq_len, batch_size, d_model] 格式
        transformer_output = self.transformer_encoder(embedded_data.permute(1, 0, 2))  # [seq_len, batch_size, d_model]

        # 还原维度 [batch_size, seq_len, output_dim]
        output = self.fc_out(transformer_output.permute(1, 0, 2))  # [batch_size, seq_len, output_dim]
        return output


# 定义判别器：用于区分真实的时间序列和生成的时间序列
class Discriminator(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 双向 LSTM 乘 2

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
    def __init__(self, generator, discriminator, gen_lr=0.00001, disc_lr=0.00001):
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # 优化器和损失函数
        self.optim_G = optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=disc_lr)
        self.criterion = nn.BCELoss()

    def train(self, real_data, condition, noise_dim, epochs=1000):
        for epoch in range(epochs):
            batch_size = real_data.size(0)

            # 训练判别器
            real_data = real_data.to(self.device)
            condition = condition.to(self.device)

            # 生成随机噪声
            noise = torch.randn(batch_size, real_data.size(1), noise_dim).to(self.device)
            fake_data = self.generator(noise, condition)

            # # 打印张量形状用于调试
            # print(f"real_data shape: {real_data.shape}")
            # print(f"condition shape: {condition.shape}")
            # print(f"noise shape: {noise.shape}")
            # print(f"fake_data shape: {fake_data.shape}")

            # 判别器预测
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            # 判别器在真实数据上的损失
            disc_real_loss = self.criterion(self.discriminator(real_data, condition), real_labels)
            # 判别器在生成数据上的损失
            disc_fake_loss = self.criterion(self.discriminator(fake_data.detach(), condition), fake_labels)
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            # 判别器反向传播
            self.optim_D.zero_grad()
            disc_loss.backward()
            self.optim_D.step()

            # 训练生成器
            gen_labels = torch.ones(batch_size, 1).to(self.device)  # 欺骗判别器，生成数据被判定为真实
            gen_loss = self.criterion(self.discriminator(fake_data, condition), gen_labels)

            # 生成器反向传播
            self.optim_G.zero_grad()
            gen_loss.backward()
            self.optim_G.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] | D Loss: {disc_loss.item()} | G Loss: {gen_loss.item()}")



# 参数定义
input_dim = 12  # 输入时间序列的维度
seq_len = 10  # 时间序列长度
d_model = 64  # Transformer 的隐藏维度
num_heads = 8  # 多头注意力头的数量
num_layers = 2  # Transformer 编码器层数
hidden_dim = 128  # 判别器的 LSTM 隐藏层维度
noise_dim = 20  # 生成器输入噪声的维度
output_dim = 12  # 生成的时间序列维度
condition_dim = 8  # 条件维度，例如特征、标签等

# 初始化生成器和判别器
generator = TransformerGenerator(input_dim + condition_dim, seq_len, d_model, num_heads, num_layers, output_dim)
discriminator = Discriminator(output_dim + condition_dim, seq_len, hidden_dim, num_layers)

# 初始化 CGAN
cgan = CGAN(generator, discriminator)

# 生成随机时间序列数据和条件
real_data = torch.randn(32, seq_len, output_dim)  # [batch_size, seq_len, output_dim]
condition = torch.randn(32, seq_len, condition_dim)  # [batch_size, seq_len, condition_dim]
# logger.info(real_data.shape)

# 训练 CGAN
cgan.train(real_data, condition, noise_dim, epochs=10000)


