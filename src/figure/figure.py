import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
from matplotlib.font_manager import FontProperties


def main():
    # 读取.jfif文件
    img_path = "../../config/R.jfif"

    # 创建一个新的绘图窗口
    fig, ax = plt.subplots()

    # 计算新的横线位置
    length = 10
    line_spacing = 1  # 每条线之间的垂直间距
    line_a_y = 4
    line_b_y = line_a_y - line_spacing
    line_c_y = line_b_y - line_spacing

    # 绘制三条横线
    a_line = plt.Line2D([0, length], [line_a_y, line_a_y], color='black')
    b_line = plt.Line2D([0, length], [line_b_y, line_b_y], color='black', linestyle='--')
    # c_line = plt.Line2D([0, 10], [line_c_y, line_c_y], color='black')
    draw_line((0, line_c_y), (2 * length / 10, line_c_y))
    draw_line((2 * length / 10, line_c_y), (8 * length / 10, line_c_y), dashed=True)
    draw_line((8 * length / 10, line_c_y), (length, line_c_y))

    # 绘制匝道
    draw_line((0, line_c_y - 1 * length / 10), (6 * length / 10, line_c_y - 1 * length / 10))
    draw_line((6 * length / 10, line_c_y - 1 * length / 10), (8 * length / 10, line_c_y))

    # 绘制匝道边界
    draw_line((8 * length / 10, line_c_y - 0.5 * length / 10), (length, line_c_y - 0.5 * length / 10),
              thickness=3)
    draw_line((6 * length / 10, line_c_y - 1.5 * length / 10), (8 * length / 10, line_c_y - 0.5 * length / 10),
              thickness=3)
    draw_line((0, line_c_y - 1.5 * length / 10), (6 * length / 10, line_c_y - 1.5 * length / 10),
              thickness=3)

    # 将横线和直角梯形添加到图形中
    ax.add_line(a_line)
    ax.add_line(b_line)

    # 添加车辆
    draw_rotated_image(img_path, (0, 1.2))
    draw_rotated_image(img_path, (2.5, 1.2))
    draw_rotated_image(img_path, (4, 1.8))
    draw_rotated_image(img_path, (6, 2.3))

    # 添加area
    area1 = Rectangle((0, line_c_y - 1 * length / 10), 2, 1, edgecolor="none", facecolor=(184/255, 219/255, 179/255),
                      label='Area1')
    area2 = Rectangle((2, line_c_y - 1 * length / 10), 4, 1, edgecolor="none", facecolor=(148/255, 198/255, 205/255),
                      label='Area2')
    traiangle_points = [(6 * length / 10, line_c_y - 1 * length / 10),
                        (6 * length / 10, line_c_y),
                        (8 * length / 10, line_c_y)]
    area3 = Polygon(traiangle_points, closed=True, edgecolor='none', facecolor=(126/255, 153/255, 244/255),
                    label='Area3')
    area4 = Rectangle((0, line_b_y - 1 * length / 10), 2, 1, edgecolor="none", facecolor=(165/255, 174/255, 183/255),
                      label='Area4')
    area5 = Rectangle((2, line_b_y - 1 * length / 10), 8, 1, edgecolor='none', facecolor=(240/255, 194/255, 132/255),
                      label='Area5')

    # 添加area到坐标轴
    ax.add_patch(area1)
    ax.add_patch(area2)
    ax.add_patch(area3)
    ax.add_patch(area4)
    ax.add_patch(area5)

    # 设置轴限制和纵横比
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    # 隐藏坐标轴
    ax.axis('off')

    # 添加图例，并设置字体为 Times New Roman
    font_prop = FontProperties(family='Times New Roman', style='normal', size=12)
    ax.legend(prop=font_prop, loc='upper left', bbox_to_anchor=(-0.35, 0.4))

    # 保存图像
    plt.savefig('../../asset/road/road.png', dpi=3000, bbox_inches='tight')

    # 显示图形
    plt.show()


def draw_line(start, end, dashed=False, thickness=1.5):
    linestyle = '--' if dashed else '-'
    x_values = [start[0], end[0]]
    y_values = [start[1], end[1]]
    plt.plot(x_values, y_values, linestyle=linestyle, color='black', linewidth=thickness)


def draw_rotated_image(image_path, point):
    # 读取.jfif文件
    img = mpimg.imread(image_path)

    # 顺时针旋转图像数组
    rotated_img = np.rot90(img, k=-1)  # 根据角度确定旋转次数

    # 绘制旋转后的图像
    plt.imshow(rotated_img, extent=[point[0], point[0] + 1.2, point[1], point[1] + 0.6], origin='lower', zorder=5)

    # 设置轴限制和纵横比
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal', adjustable='box')


if __name__ == '__main__':
    main()

