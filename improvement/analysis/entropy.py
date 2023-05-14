import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


'''
计算图像的信息熵
'''


def _entropy(img):
    img = cv2.imread(img)
    w, h, _ = img.shape
    B, G, R = cv2.split(img)
    gray, num1 = np.unique(R, return_counts=True)
    gray, num2 = np.unique(G, return_counts=True)
    gray, num3 = np.unique(B, return_counts=True)
    R_entropy = 0
    G_entropy = 0
    B_entropy = 0

    for i in range(len(gray)):
        p1 = num1[i]/(w*h)
        p2 = num2[i]/(w*h)
        p3 = num3[i]/(w*h)
        R_entropy -= p1*(math.log(p1, 2))
        G_entropy -= p2*(math.log(p2, 2))
        B_entropy -= p3*(math.log(p3, 2))
    return R_entropy, G_entropy, B_entropy


def entropy(raw_img, encrypt_img):
    with open('result.txt','a+', encoding='utf8') as f:
        R_entropy, G_entropy, B_entropy = _entropy(raw_img)
        f.write('====原图像信息熵====\n')
        f.write('通道R:\t\t{:.5}\n'.format(R_entropy))
        f.write('通道G:\t\t{:.5}\n'.format(G_entropy))
        f.write('通道B:\t\t{:.5}\n'.format(B_entropy))
        f.write('\n')
        R_entropy, G_entropy, B_entropy = _entropy(encrypt_img)
        f.write('===加密图像信息熵===\n')
        f.write('通道R:\t\t{:.5}\n'.format(R_entropy))
        f.write('通道G:\t\t{:.5}\n'.format(G_entropy))
        f.write('通道B:\t\t{:.5}\n'.format(B_entropy))
        f.write('\n')


if __name__ == '__main__':
    img = '../images/lena.png'
    encrypt_img = '../images/lena_encrypt.png'
    entropy(img, encrypt_img)
