# 原代码地址：https://blog.csdn.net/qq_41137110/article/details/115675014

import cv2
import numpy as np
import matplotlib.pyplot as plt
from encrypt import encrypt
from os import remove

'''
计算像素数变化率
'''


def NPCR(img1, img2):
    w, h, = img1.shape

    # 图像通道拆分
    # 返回数组的排序后的唯一元素和每个元素重复的次数
    ar, num = np.unique((img1 != img2), return_counts=True)
    npcr = (num[0] if ar[0] == True else num[1])/(w*h)

    return npcr


'''
两张图像之间的平均变化强度
'''


def UACI(img1, img2):
    h, w = img1.shape
    # 元素为uint8类型取值范围：0到255
    # 强制转换元素类型，为了运算
    img1 = img1.astype(np.int16)
    img2 = img2.astype(np.int16)

    sum = np.sum(abs(img1-img2))
    uaci = sum/255/(w*h)

    return uaci


def differencial(img_path):
    img_path_2 = img_path.rsplit('.', 1)[0] + '_differencial.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img[0][0] = img[0][0] ^ 2**4  # 修改第一个像素值的第5位
    cv2.imwrite(img_path_2, img)
    _, img1 = encrypt(img_path)
    _, img2 = encrypt(img_path_2)
    npcr = NPCR(img1, img2)
    print('=================差分攻击=================')
    # 百分数表示，保留小数点后4位
    print('PSNR:\t{:.4%}'.format(npcr))

    uaci = UACI(img1, img2)
    # 百分数表示，保留小数点后4位
    print('UACI:\t{:.4%}'.format(uaci))
    remove(img_path_2)
    remove(img_path_2.rsplit('.', 1)[0] + '_encrypt.png')


if __name__ == '__main__':
    img_path = '../images/lena.png'
    differencial(img_path)
