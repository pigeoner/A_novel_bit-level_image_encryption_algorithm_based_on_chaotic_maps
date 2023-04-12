# 原代码地址：https://blog.csdn.net/qq_41137110/article/details/115675014

import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
分别计算图像通道相邻像素的水平、垂直和对角线的相关系数并返回
'''


def _correlation(img, N):
    # 计算img通道
    h, w = img.shape
    # 随机产生pixels个[0,w-1)范围内的整数序列
    row = np.random.randint(0, h-1, N)
    col = np.random.randint(0, w-1, N)
    # 绘制相邻像素相关性图,统计x,y坐标
    x = []
    h_y = []
    v_y = []
    d_y = []
    for i in range(N):
        # 选择当前一个像素
        x.append(img[row[i]][col[i]])
        # 水平相邻像素是它的右侧也就是同行下一列的像素
        h_y.append(img[row[i]][col[i]+1])
        # 垂直相邻像素是它的下方也就是同列下一行的像素
        v_y.append(img[row[i]+1][col[i]])
        # 对角线相邻像素是它的右下即下一行下一列的那个像素
        d_y.append(img[row[i]+1][col[i]+1])
    # 三个方向的合到一起
    x = x*3
    y = h_y+v_y+d_y

    # 计算E(x)，计算三个方向相关性时，x没有重新选择也可以更改
    ex = 0
    for i in range(N):
        ex += img[row[i]][col[i]]
    ex = ex/N
    # 计算D(x)
    dx = 0
    for i in range(N):
        dx += (img[row[i]][col[i]]-ex)**2
    dx /= N

    # 水平相邻像素h_y
    # 计算E(y)
    h_ey = 0
    for i in range(N):
        h_ey += img[row[i]][col[i]+1]
    h_ey /= N
    # 计算D(y)
    h_dy = 0
    for i in range(N):
        h_dy += (img[row[i]][col[i]+1]-h_ey)**2
    h_dy /= N
    # 计算协方差
    h_cov = 0
    for i in range(N):
        h_cov += (img[row[i]][col[i]]-ex)*(img[row[i]][col[i]+1]-h_ey)
    h_cov /= N
    h_Rxy = h_cov/(np.sqrt(dx)*np.sqrt(h_dy))

    # 垂直相邻像素v_y
    # 计算E(y)
    v_ey = 0
    for i in range(N):
        v_ey += img[row[i]+1][col[i]]
    v_ey /= N
    # 计算D(y)
    v_dy = 0
    for i in range(N):
        v_dy += (img[row[i]+1][col[i]]-v_ey)**2
    v_dy /= N
    # 计算协方差
    v_cov = 0
    for i in range(N):
        v_cov += (img[row[i]][col[i]]-ex)*(img[row[i]+1][col[i]]-v_ey)
    v_cov /= N
    v_Rxy = v_cov/(np.sqrt(dx)*np.sqrt(v_dy))

    # 对角线相邻像素d_y
    # 计算E(y)
    d_ey = 0
    for i in range(N):
        d_ey += img[row[i]+1][col[i]+1]
    d_ey /= N
    # 计算D(y)
    d_dy = 0
    for i in range(N):
        d_dy += (img[row[i]+1][col[i]+1]-d_ey)**2
    d_dy /= N
    # 计算协方差
    d_cov = 0
    for i in range(N):
        d_cov += (img[row[i]][col[i]]-ex) * \
            (img[row[i]+1][col[i]+1]-d_ey)
    d_cov /= N
    d_Rxy = d_cov/(np.sqrt(dx)*np.sqrt(d_dy))

    return h_Rxy, v_Rxy, d_Rxy, x, y


'''
分别计算图像img的各通道相邻像素的相关系数，默认随机选取2000对相邻像素
'''


def show_correlation(raw_img, encrypt_img, N=2000):
    r_img = cv2.imread(raw_img, cv2.IMREAD_GRAYSCALE)
    e_img = cv2.imread(encrypt_img, cv2.IMREAD_GRAYSCALE)
    r_Rxy = _correlation(r_img, N)
    e_Rxy = _correlation(e_img, N)

    # 结果展示
    # 原图像
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(221)
    plt.imshow(r_img, cmap='gray')
    plt.title('原图像')
    # 原图像相邻像素分布
    plt.subplot(222)
    plt.scatter(r_Rxy[3], r_Rxy[4], s=1)
    plt.title('像素分布')
    # 加密图像
    plt.subplot(223)
    plt.imshow(e_img, cmap='gray')
    plt.title('加密图像')
    # 加密图像相邻像素分布
    plt.subplot(224)
    plt.scatter(e_Rxy[3], e_Rxy[4], s=1)
    plt.title('像素分布')
    plt.show()

    return r_Rxy[0:3], e_Rxy[0:3]


def correlation(img, encrypt_img):
    r_Rxy, e_Rxy = show_correlation(img, encrypt_img)
    # 输出结果保留四位有效数字
    print("========原始图像的各方向的相关系数========")
    print('Horizontal\tVertical\tDiagonal')
    print('{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(
        r_Rxy[0], r_Rxy[1], r_Rxy[2]))
    print("========加密图像的各方向的相关系数========")
    print('Horizontal\tVertical\tDiagonal')
    print('{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(
        e_Rxy[0], e_Rxy[1], e_Rxy[2]))


if __name__ == '__main__':
    img = '../images/lena.png'
    encrypt_img = '../images/lena_encrypt.png'
    correlation(img, encrypt_img)
