import cv2
from matplotlib import pyplot as plt
from encrypt import encrypt
from decrypt import decrypt
import numpy as np
from copy import deepcopy


def encrypt_sensitivity(img_path, key=None, modified_key=None):
    key = [0.01234567890123, 0.12345678912345,
           0.01234567891234, 0.21234567891234] if not key else key
    modified_key = [0.01234567890124, 0.12345678912345,
                    0.01234567891234, 0.21234567891234] if not modified_key else modified_key

    # 结果展示
    # 原图像
    r_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(221)
    plt.imshow(r_img, cmap='gray')
    plt.title('原图像')

    # key加密的图像
    _, key_img = encrypt(img_path, *key)
    plt.subplot(222)
    plt.imshow(key_img, cmap='gray')
    plt.title('key加密的图像')

    # 修改过的key加密的图像
    _, modified_key_img = encrypt(img_path, *modified_key)
    plt.subplot(223)
    plt.imshow(modified_key_img, cmap='gray')
    plt.title('修改过的key加密的图像')

    # 两者差值
    plt.subplot(224)
    plt.imshow(np.abs(modified_key_img - key_img), cmap='gray')
    plt.title('两者差值')
    plt.show()


def decrypt_sensitivity(img_path, encrypt_img_path, key=None, incorrect_key=None):
    key = [0.01234567890123, 0.12345678912345,
           0.01234567891234, 0.21234567891234] if not key else key
    if not incorrect_key:
        incorrect_key = deepcopy(key)
        incorrect_key[0] += 0.00000000000001

    # 结果展示
    # 原图像
    r_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(221)
    plt.imshow(r_img, cmap='gray')
    plt.title('原图像')

    # 原始key加密的图像
    _, key_img = encrypt(img_path, *key)
    plt.subplot(222)
    plt.imshow(key_img, cmap='gray')
    plt.title('原始key加密的图像')

    # 修改过的key解密的图像
    # 读取参数
    raw_params = np.load('./params/params.npz')
    np.savez('./params/incorrect_params.npz',
             x0=incorrect_key[0], u1=incorrect_key[1],
             y0=incorrect_key[2], u2=incorrect_key[3],
             A11_0=raw_params['A11_0'], A22_0=raw_params['A22_0'],
             n_round=raw_params['n_round'])
    incorrect_decrypt_img = decrypt(
        encrypt_img_path, params_path='incorrect_params.npz')
    plt.subplot(223)
    plt.imshow(incorrect_decrypt_img, cmap='gray')
    plt.title('修改过的key解密的图像')

    # 原始key解密的图像
    decrypt_img = decrypt(encrypt_img_path, params_path='params.npz')
    plt.subplot(224)
    plt.imshow(decrypt_img, cmap='gray')
    plt.title('原始key解密的图像')
    plt.show()


if __name__ == "__main__":
    img_path = "../images/lena.png"
    encrypt_img_path = "../images/lena_encrypt.png"
    encrypt_sensitivity(img_path)
    decrypt_sensitivity(img_path, encrypt_img_path, None, None)
