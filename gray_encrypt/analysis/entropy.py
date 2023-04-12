# 原代码地址：https://blog.csdn.net/qq_41137110/article/details/115675014

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


'''
计算图像的信息熵
'''


def _entropy(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    gray, num = np.unique(img, return_counts=True)
    entropy = 0

    for i in range(len(gray)):
        p = num[i]/(w*h)
        entropy -= p*(math.log(p, 2))
    return entropy


def entropy(raw_img, encrypt_img):
    # 图像lena的熵
    raw_entropy = _entropy(raw_img)
    encrypt_entropy = _entropy(encrypt_img)
    print('==================信息熵==================')
    print('原图像: \t{:.5}'.format(raw_entropy))
    print('加密图像: \t{:.5}'.format(encrypt_entropy))


if __name__ == '__main__':
    img = '../images/lena.png'
    encrypt_img = '../images/lena_encrypt.png'
    entropy(img, encrypt_img)
