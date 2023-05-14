import cv2
from numpy import uint8, zeros, sin, pi, cos
from random import uniform


def img_bit_decomposition(img):
    '''图像的位平面分解'''
    m, n = img.shape
    r = zeros((8, m, n), dtype=uint8)
    for i in range(8):
        r[i, :, :] = cv2.bitwise_and(img, 2**i)
        mask = r[i, :, :] > 0
        r[i, mask] = 1
    return r


def PWLCM(x0, h, u, num):
    '''生成混沌映射序列
    x0: 初始值
    h: 控制参数
    num: 生成序列的长度
    '''
    pwlcm = [0] * num
    pwlcm[0] = x0
    for i in range(1, num):
        xn = pwlcm[i - 1]
        if xn > 0 and xn < h:
            pwlcm[i] = (xn/h+u*sin(u*(pi*xn))) % 1
        elif xn >= h and xn < 0.5:
            pwlcm[i] = ((xn-h)/(0.5-h)+u*sin(u*(pi*xn))) % 1
        elif xn >= 0.5 and xn < 1 - h:
            pwlcm[i] = ((1-xn-h)/(0.5-h)+u*sin(u*(pi*(1-xn)))) % 1
        elif xn >= 1 - h and xn < 1:
            pwlcm[i] = ((1-xn)/h+u*sin(u*(pi*(1-xn)))) % 1
        else:
            raise ValueError("xi must be in [0, 1]")
    return pwlcm
