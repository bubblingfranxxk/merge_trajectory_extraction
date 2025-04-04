# -*- coding = utf-8 -*-
# @Time : 2025/3/31 16:20
# @Author : 王砚轩
# @File : loss_plot.py
# @Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# 设置Seaborn的样式（可选）
sns.set(style="whitegrid")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

rootPath = os.path.abspath('../../')
assetPath = rootPath + '/asset/'
# 读取CSV文件，替换为你的文件路径
file_path = assetPath + 'temp2.csv'
df = pd.read_csv(file_path)

# 创建图形
plt.figure(figsize=(12, 6))

# 使用tab10调色板
colors = sns.color_palette("tab10", n_colors=len(df.columns))

# 为每一列绘制折线图（使用sns.lineplot，并指定x轴为0~249的序号）
for idx, column in enumerate(df.columns):
    sns.lineplot(x=range(len(df)), y=df[column],
                 label=column,
                 color=colors[idx],
                 # linestyle='--',  # 虚线样式
                 linewidth=3)  # 线宽较粗

# 添加标签和标题
plt.xlabel('Epoch')
plt.ylabel('JS-Divergence')
plt.title('JS散度折线图')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=16)  # 显示图例
plt.grid(True)

# 自动调整子图布局
plt.tight_layout()

# 保存高清图片（替换为你的保存路径）
save_path = assetPath+'Model_JS.png'  # 支持.jpg/.pdf/.svg等格式
plt.savefig(
    save_path,
    dpi=500,                 # 设置分辨率
    bbox_inches='tight',     # 自动裁剪空白区域
    transparent=False        # 背景透明可选
)

# 显示图形
plt.show()