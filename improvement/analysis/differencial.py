import cv2
import numpy as np
from encrypt import encrypt
from os import remove

'''
计算像素数变化率
'''


def NPCR(img1, img2):
    # opencv颜色通道顺序为BGR
    w, h, _ = img1.shape

    # 图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 返回数组的排序后的唯一元素和每个元素重复的次数
    ar, num = np.unique((R1 != R2), return_counts=True)
    R_npcr = (num[0] if ar[0] == True else num[1])/(w*h)
    ar, num = np.unique((G1 != G2), return_counts=True)
    G_npcr = (num[0] if ar[0] == True else num[1])/(w*h)
    ar, num = np.unique((B1 != B2), return_counts=True)
    B_npcr = (num[0] if ar[0] == True else num[1])/(w*h)

    return R_npcr, G_npcr, B_npcr


'''
两张图像之间的平均变化强度
'''


def UACI(img1, img2):
    w, h, _ = img1.shape
    # 图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 元素为uint8类型取值范围：0到255
    # print(R1.dtype)

    # 强制转换元素类型，为了运算
    R1 = R1.astype(np.int16)
    R2 = R2.astype(np.int16)
    G1 = G1.astype(np.int16)
    G2 = G2.astype(np.int16)
    B1 = B1.astype(np.int16)
    B2 = B2.astype(np.int16)

    sumR = np.sum(abs(R1-R2))
    sumG = np.sum(abs(G1-G2))
    sumB = np.sum(abs(B1-B2))
    R_uaci = sumR/255/(w*h)
    G_uaci = sumG/255/(w*h)
    B_uaci = sumB/255/(w*h)

    return R_uaci, G_uaci, B_uaci


def differencial(img_path, ntimes=1):
    img_path_2 = img_path.rsplit('.', 1)[0] + '_differencial.png'
    img = cv2.imread(img_path)
    img[0][0][0] = img[0][0][0] ^ 1  # 修改B通道第一个像素值的第5位
    cv2.imwrite(img_path_2, img)
    _, img1 = encrypt(img_path)
    _, img2 = encrypt(img_path_2)
    R_npcr_mean = []
    G_npcr_mean = []
    B_npcr_mean = []
    for i in range(ntimes):
        R_npcr, G_npcr, B_npcr = NPCR(img1, img2)
        R_npcr_mean.append(R_npcr)
        G_npcr_mean.append(G_npcr)
        B_npcr_mean.append(B_npcr)
    # print('\n   *差分攻击*   ')
    with open('result.txt','a+', encoding='utf8') as f:
        f.write('========NPCR========\n')
        # 百分数表示，保留小数点后4位
        f.write('Red  :\t{:.4%}\n'.format(np.mean(R_npcr_mean)))
        f.write('Green:\t{:.4%}\n'.format(np.mean(G_npcr_mean)))
        f.write('Blue :\t{:.4%}\n'.format(np.mean(B_npcr_mean)))
        f.write('\n')

    R_uaci_mean = []
    G_uaci_mean = []
    B_uaci_mean = []
    for i in range(100):
        R_uaci, G_uaci, B_uaci = UACI(img1, img2)
        R_uaci_mean.append(R_uaci)
        G_uaci_mean.append(G_uaci)
        B_uaci_mean.append(B_uaci)
    with open('result.txt', 'a+', encoding='utf8') as f:
        f.write('========UACI========\n')
        # 百分数表示，保留小数点后4位
        f.write('Red  :\t{:.4%}\n'.format(np.mean(R_uaci_mean)))
        f.write('Green:\t{:.4%}\n'.format(np.mean(G_uaci_mean)))
        f.write('Blue :\t{:.4%}\n'.format(np.mean(B_uaci_mean)))
        f.write('\n')
    remove(img_path_2)
    remove(img_path_2.rsplit('.', 1)[0] + '_encrypt.png')


if __name__ == '__main__':
    img_path = '../images/lena.png'
    differencial(img_path, ntimes=100)
