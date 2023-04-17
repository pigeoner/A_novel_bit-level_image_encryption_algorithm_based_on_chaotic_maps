import cv2
from numpy import uint8, zeros


def img_bit_decomposition(img):
    '''图像的位平面分解'''
    m, n = img.shape
    r = zeros((8, m, n), dtype=uint8)
    for i in range(8):
        r[i, :, :] = cv2.bitwise_and(img, 2**i)
        mask = r[i, :, :] > 0
        r[i, mask] = 1
    return r


def PWLCM(x0, h, num):
    '''生成混沌映射序列
    x0: 初始值
    h: 控制参数
    num: 生成序列的长度
    '''
    pwlcm = [0] * num
    pwlcm[0] = x0
    for i in range(1, num):
        if pwlcm[i - 1] > 0 and pwlcm[i - 1] < h:
            pwlcm[i] = pwlcm[i - 1] / h
        elif pwlcm[i - 1] >= h and pwlcm[i - 1] < 0.5:
            pwlcm[i] = (pwlcm[i - 1] - h) / (0.5 - h)
        elif pwlcm[i - 1] >= 0.5 and pwlcm[i - 1] < 1 - h:
            pwlcm[i] = (1 - pwlcm[i - 1] - h) / (0.5 - h)
        elif pwlcm[i - 1] >= 1 - h and pwlcm[i - 1] < 1:
            pwlcm[i] = (1 - pwlcm[i - 1]) / h
        else:
            raise ValueError("xi must be in [0, 1]")
    return pwlcm
