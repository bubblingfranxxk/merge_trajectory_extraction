# -*- coding = utf-8 -*-
# @Time : 2024/6/2 17:05
# @Author : 王砚轩
# @File : test.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw, dtw_visualisation as dtwvis

# 定义两个时间序列
series1 = np.array([1, 2, 3, 4, 2, 3, 4, 5])
series2 = np.array([2, 3, 4, 3, 2, 1, 2, 3])

# 计算 DTW 距离
distance = dtw.distance(series1, series2)
print(f"DTW 距离: {distance}")

# 可视化对齐结果
path = dtw.warping_path(series1, series2)
dtwvis.plot_warping(series1, series2, path)
plt.show()

