from math import floor
import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from utils import PWLCM, img_bit_decomposition


def decrypt(encrypt_img_path, raw_img_path=None, params_path='params.npz', show=False):
    '''
    解密图像
    '''
    npz_file = np.load(f"./params/{params_path}")  # 加载参数
    x0, u1, y0, u2, A11_0, A22_0, n_round = \
        npz_file['x0'], \
        npz_file['u1'], \
        npz_file['y0'], \
        npz_file['u2'], \
        npz_file['A11_0'], \
        npz_file['A22_0'], \
        npz_file['n_round']
    A11_0 = [int(i) for i in list(bin(A11_0)[2:])[1:]]
    A11_0 = [A11_0[2*len(A11_0)//3:], A11_0[len(A11_0)//3:2 *
                                            len(A11_0)//3], A11_0[:len(A11_0)//3]]
    A22_0 = [int(i) for i in list(bin(A22_0)[2:])[1:]]
    A22_0 = [A22_0[2*len(A22_0)//3:], A22_0[len(A22_0)//3:2 *
                                            len(A22_0)//3], A22_0[:len(A22_0)//3]]
    # print(
    #     f"参数:\nx0: {x0}, u1: {u1},\ny0: {y0}, u2: {u2},\nA11_0: {A11_0}, A22_0: {A22_0},\nn_round: {n_round}"
    # )
    filename, ext = encrypt_img_path.rsplit('.', 1)
    decrypt_img_path = f"{filename}_decrypt.{ext}"

    img = cv2.imread(encrypt_img_path)  # 读取图像

    M, N, _ = img.shape  # M: height, N: width

    B, G, R = cv2.split(img)  # BGR

    A = []

    if M != M or N != N:
        raise ValueError("The shape of image is not correct!")

    if show:
        print(f"图像分辨率为: {N} x {M}")

    N0 = 1000  # 舍弃前N0个数
    for ib, b in enumerate([B, G, R]):
        C = b
        # 逆双向扩散
        z0 = ((y0 + M*N) / 256) % 1  # key3的初始值z0
        u3 = u2  # key3的参数u3
        Z_raw = PWLCM(z0, u3, N0 + 2 * M * N)[N0:]  # 生成长度为N0+2*M*N的混沌序列
        Z = [floor(z * 1e14) % 256 for z in Z_raw]  # 标准化
        Z1, Z2 = Z[:M * N], Z[M * N:]
        V = np.rot90(C, 2).T.ravel()
        for i in range(M*N - 1, 0, -1):
            V[i] = V[i] ^ V[i - 1] ^ Z2[i]
        V[0] = V[0] ^ (M*N) ^ Z2[0]  # 第一个像素点的解密
        H = np.rot90(V.reshape(M, N).T, -2).ravel()
        for i in range(M*N - 1, 0, -1):
            H[i] = H[i] ^ H[i - 1] ^ Z1[i]
        H[0] = H[0] ^ (M*N) ^ Z1[0]  # 第一个像素点的解密
        C = H.reshape(M, N)

        PWLCM_MAP_X = PWLCM(x0, u1, N0 + M*N)[N0:]
        X1 = [floor(i * 1e14) % 256 for i in PWLCM_MAP_X]  # key1
        X1_reshape = np.mat(X1, dtype=np.uint8).reshape(M, N)
        X1_bitplanes = img_bit_decomposition(X1_reshape)  # key1的位平面分解
        b1 = X1_bitplanes[::2, :, :].ravel()  # key1的偶数位平面
        b2 = X1_bitplanes[1::2, :, :].ravel()  # key1的奇数位平面

        bitplanes = img_bit_decomposition(C)  # 位平面分解
        C1 = bitplanes[:4, :, :].ravel()
        C2 = bitplanes[4:, :, :].ravel()

        L = M * N * 4
        for k in range(n_round):
            # 逆混淆
            sum = np.sum(C1) + np.sum(C2)  # C1 和 C2 的和
            s0 = (y0 + sum / L) % 1  # key2的初始值s0
            S = PWLCM(s0, u2, N0 + 2 * L)[N0:]
            S1, S2 = S[:L], S[L:]
            Y = [floor(s1 * 1e14) % L for s1 in S1]
            Z = [floor(s2 * 1e14) % L for s2 in S2]

            for j in range(L-1, -1, -1):
                C2[j], C1[Z[j]] = C1[Z[j]], C2[j]
            for i in range(L-1, -1, -1):
                C1[i], C2[Y[i]] = C2[Y[i]], C1[i]

            B1, B2 = C1, C2

            # 逆扩散
            sum2 = np.sum(B1)  # B1 的和
            A11 = np.zeros(L, dtype=np.uint8)
            A22 = np.zeros(L, dtype=np.uint8)
            A22[0] = A22_0[ib][k]
            for i in range(1, L):
                A22[i] = A22[i - 1] ^ B1[i] ^ B2[i] ^ b2[i]
            A2 = np.roll(A22, -1*sum2)  # A2 循环左移 sum2 位
            sum1 = np.sum(A2)  # A2 的和
            A11[0] = A11_0[ib][k]
            for i in range(1, L):
                A11[i] = A11[i - 1] ^ A2[i] ^ B1[i] ^ b1[i]
            A1 = np.roll(A11, -1*sum1)  # A1 循环左移 sum1 位
            C1 = deepcopy(A1)
            C2 = deepcopy(A2)
        A.append([A1, A2])

    A[0][1], A[1][1] = A[1][1], A[0][1]
    A[1][1], A[2][1] = A[2][1], A[1][1]

    A_BGR = []
    for e in A:
        A_IMG = np.append(e[0], e[1])[::-1].reshape(8, M, N)[:, ::-1, ::-1]
        res = np.zeros((M, N), dtype=np.uint8)
        for i in range(8):
            res = cv2.bitwise_or(res, A_IMG[i, :, :] << i)
        A_BGR.append(res)
    I = cv2.merge(A_BGR[::-1])

    cv2.imwrite(decrypt_img_path, I)

    # 验证解密得到的图像为原图像
    if raw_img_path:
        raw_img = cv2.imread(raw_img_path)
        r_B, r_G, r_R = cv2.split(raw_img)
        R, G, B = A_BGR
        if not np.sum(np.abs(r_B-B)) and not np.sum(np.abs(r_G-G)) and not np.sum(np.abs(r_R-R)):
            print("解密成功!")
        else:
            print("解密失败!")

    # 显示图像
    if show:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
        plt.imshow(cv2.merge(A_BGR))
        plt.title("解密图像")
        plt.show()

    return I


if __name__ == "__main__":
    encrypt_img_path = "./data/rabbit1_encrypt.png"  # 要解密的图像路径
    raw_img_path = "./data/rabbit1.jpg"  # 原始图像路径
    decrypt(encrypt_img_path, raw_img_path)
