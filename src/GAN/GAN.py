# -*- coding = utf-8 -*-
# @Time : 2024/5/19 15:42
# @Author : 王砚轩
# @File : GAN.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建输出目录用于保存生成的图像
os.makedirs("images", exist_ok=True)

# 超参数设置
latent_dim = 100  # 潜在空间维度，即生成器的输入噪声向量的维度
data_dim = 16  # 数据维度，我们将生成16维数据
batch_size = 128  # 每个批次的样本数量
epochs = 100  # 训练的轮数
learning_rate = 0.0002  # 学习率
b1 = 0.5  # Adam优化器的一阶矩估计的衰减率
b2 = 0.999  # Adam优化器的二阶矩估计的衰减率


# 随机生成一些真实数据
def generate_real_data(samples):
    return np.random.normal(0, 1, (samples, data_dim))


# 构建生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128, 0.8),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, 0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, 0.8),
            nn.Linear(512, data_dim),
            nn.Tanh()
        )

    def forward(self, z):
        data = self.model(z)
        return data


# 构建判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        validity = self.model(data)
        return validity


# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 损失函数
adversarial_loss = nn.BCELoss()

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

# 准备数据
real_data = generate_real_data(10000)  # 生成10000个样本的真实数据
dataset = TensorDataset(torch.tensor(real_data, dtype=torch.float32))  # 创建TensorDataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 创建DataLoader


# 训练GAN模型
def train_gan():
    for epoch in range(epochs):
        for i, (real_imgs,) in enumerate(dataloader):
            # 真实图像标签为1，生成图像标签为0
            valid = torch.ones(real_imgs.size(0), 1, requires_grad=False)
            fake = torch.zeros(real_imgs.size(0), 1, requires_grad=False)

            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(real_imgs.size(0), latent_dim)  # 生成潜在空间中的随机噪声
            gen_data = generator(z)  # 生成器生成假数据
            g_loss = adversarial_loss(discriminator(gen_data), valid)  # 生成器的损失函数
            g_loss.backward()  # 反向传播计算梯度
            optimizer_G.step()  # 更新生成器参数

            # 训练判别器
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)  # 判别器对真实数据的损失
            fake_loss = adversarial_loss(discriminator(gen_data.detach()), fake)  # 判别器对假数据的损失
            d_loss = (real_loss + fake_loss) / 2  # 判别器的总损失
            d_loss.backward()  # 反向传播计算梯度
            optimizer_D.step()  # 更新判别器参数

            # 打印进度
            print(
                f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # 保存生成的数据样本
        if epoch % 10 == 0:
            save_data = gen_data.data[:25].cpu().numpy()  # 获取生成的前25个样本
            fig, axs = plt.subplots(5, 5, figsize=(10, 10))  # 创建一个5x5的子图网格
            cnt = 0
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(save_data[cnt].reshape(4, 4), cmap='gray')  # 显示生成的数据
                    axs[i, j].axis('off')  # 关闭坐标轴
                    cnt += 1
            fig.savefig(f"images/generated_data_{epoch}.png")  # 保存图像
            plt.close()  # 关闭图像


# GAN模型的预测
# 使用训练好的生成器生成新的数据
def generate_samples(generator, num_samples):
    z = torch.randn(num_samples, latent_dim)  # 生成潜在空间中的随机噪声
    gen_data = generator(z)  # 生成器生成假数据
    return gen_data


# 生成一些样本
def main():
    train_gan()
    num_samples = 10
    generated_samples = generate_samples(generator, num_samples)
    print(generated_samples)


# 使用main函数作为入口点
if __name__ == "__main__":
    main()
