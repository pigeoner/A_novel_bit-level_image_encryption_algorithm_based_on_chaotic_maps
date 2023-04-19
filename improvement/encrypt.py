from math import floor
import cv2
import numpy as np
import os
from random import uniform
from copy import deepcopy
from matplotlib import pyplot as plt
from functools import reduce
from utils import img_bit_decomposition, PWLCM


def encrypt(
    img_path,
    x0=None, u1=None, y0=None, u2=None,  # 密钥
    n_round=1,  # 加密轮数
    params_path='params.npz',  # 参数保存路径
    show=False,  # 是否显示图像
    use_params=False  # 是否使用已有参数
):
    '''
    加密图像
    x0, u1, y0, u2: 混沌序列的初始值和参数
    N0: 丢弃前N0个数
    n_round: 加密轮数
    params_path: 参数保存路径，只需要写文件名+后缀，不需要写绝对路径
    show: 是否显示图像
    use_params: 是否使用已有参数
    '''
    if not use_params:
        x0 = uniform(1e-16, 1-1e-16) if not x0 else x0  # key1的初始值x0
        u1 = uniform(1e-16, 0.5-1e-16) if not u1 else u1  # key1的初始值u1
        y0 = uniform(1e-16, 1-1e-16) if not y0 else y0  # 初始值y0
        u2 = uniform(1e-16, 0.5-1e-16) if not u2 else u2  # 初始值u2
    else:
        use_params_path = f"./params/{params_path}"
        if os.path.exists(use_params_path):
            params = np.load(use_params_path)
            x0 = params['x0']
            u1 = params['u1']
            y0 = params['y0']
            u2 = params['u2']
            n_round = params['n_round']
        else:
            raise FileNotFoundError(f"未找到参数文件: {use_params_path}")
    filename, ext = img_path.rsplit('.', 1)
    encrypt_img_path = f"{filename}_encrypt.png"

    img = cv2.imread(img_path)  # 读取图像

    M, N, _ = img.shape  # M: height, N: width

    B, G, R = cv2.split(img)

    if show:
        print(f"图像分辨率为: {N} x {M}")

    B_bitplanes = img_bit_decomposition(B)[::-1, :, :]  # 位平面分解
    G_bitplanes = img_bit_decomposition(G)[::-1, :, :]
    R_bitplanes = img_bit_decomposition(R)[::-1, :, :]
    B_A1 = B_bitplanes[:4, :, :].ravel()  # B的高位平面
    B_A2 = B_bitplanes[4:, :, :].ravel()  # B的低位平面
    G_A1 = G_bitplanes[:4, :, :].ravel()  # G的高位平面
    G_A2 = G_bitplanes[4:, :, :].ravel()  # G的低位平面
    R_A1 = R_bitplanes[:4, :, :].ravel()  # R的高位平面
    R_A2 = R_bitplanes[4:, :, :].ravel()  # R的低位平面
    B_A2, G_A2 = G_A2, B_A2  # 交换B和G的低位平面
    G_A2, R_A2 = R_A2, G_A2  # 交换G和R的低位平面
    E_BGR = []
    A11_0 = [[], [], []]
    A22_0 = [[], [], []]
    for ib, b in enumerate([(B_A1, B_A2), (G_A1, G_A2), (R_A1, R_A2)]):
        A1, A2 = b
        # Piecewise Logistic Chaotic Map
        N0 = 1000  # 丢弃前N0个数
        PWLCM_MAP_X = PWLCM(x0, u1, N0 + M*N)[N0:]  # 生成长度为N0+M*N的混沌序列
        # key1，将[PWLCM_MAP_X]中每个小数转换为整数并对256取模得到0~255之间的整数作为key1的值
        X1 = [floor(i * 1e14) % 256 for i in PWLCM_MAP_X]
        X1_reshape = np.mat(X1, dtype=np.uint8).reshape(M, N)
        X1_bitplanes = img_bit_decomposition(X1_reshape)  # key1的位平面分解
        b1 = X1_bitplanes[::2, :, :].ravel()  # key1的偶数位平面
        b2 = X1_bitplanes[1::2, :, :].ravel()  # key1的奇数位平面

        L = M * N * 4
        for k in range(n_round):
            # 初始化B1和B2
            B1 = np.zeros(L, dtype=np.uint8)
            B2 = np.zeros(L, dtype=np.uint8)
            # 扩散
            sum1 = np.sum(A2)  # A2 的和
            A11 = np.roll(A1, sum1)  # 高位平面的循环右移 sum1 位
            for i in range(L):
                B1[i] = A11[i] ^ A11[i - 1] ^ A2[i] ^ b1[i]
            sum2 = np.sum(B1)  # B1 的和
            A22 = np.roll(A2, sum2)  # 低位平面的循环右移 sum2 位
            for i in range(L):
                B2[i] = A22[i] ^ A22[i - 1] ^ B1[i] ^ b2[i]

            # 生成 Y 和 Z
            sum = np.sum(B1) + np.sum(B2)

            s0 = (y0 + sum / L) % 1  # key2的初始值s0
            S = PWLCM(s0, u2, N0 + 2 * L)[N0:]
            S1, S2 = S[:L], S[L:]
            # 将S1中每个小数转换为整数并对L取模得到0~L-1之间的整数作为Y的值
            Y = [floor(s1 * 1e14) % L for s1 in S1]
            # 将S2中每个小数转换为整数并对L取模得到0~L-1之间的整数作为Z的值
            Z = [floor(s2 * 1e14) % L for s2 in S2]

            # 混淆
            for i in range(L):
                B1[i], B2[Y[i]] = B2[Y[i]], B1[i]
            for j in range(L):
                B2[j], B1[Z[j]] = B1[Z[j]], B2[j]
            A11_0[ib].append(A11[0])
            A22_0[ib].append(A22[0])
            # 用B1和B2更新A1和A2，以作为下一轮加密的输入
            A1 = deepcopy(B1)
            A2 = deepcopy(B2)
        # C1，C2 序列合并成加密图像
        C_bitplanes = np.append(B1, B2).reshape(8, M, N)
        C = np.zeros((M, N), dtype=np.uint8)
        for i in range(8):
            C = cv2.bitwise_or(C, C_bitplanes[i, :, :] << i)

        # 像素级双向扩散
        z0 = ((y0 + M*N) / 256) % 1  # key3的初始值z0
        u3 = u2  # key3的参数u3
        Z_raw = PWLCM(z0, u3, N0 + 2 * M * N)[N0:]  # 生成长度为N0+2*M*N的混沌序列
        # 标准化，将Z_raw中每个小数转换为整数并对256取模得到0~255之间的整数作为Z的值
        Z = [floor(z * 1e14) % 256 for z in Z_raw]
        Z1, Z2 = Z[:M * N], Z[M * N:]
        H = C.ravel()  # 横向展开
        H[0] = H[0] ^ (M*N) ^ Z1[0]  # 第一个像素点的加密
        for i in range(1, M*N):
            H[i] = H[i] ^ H[i - 1] ^ Z1[i]
        V = np.rot90(H.reshape(M, N), 2).T.ravel()  # 纵向展开
        V[0] = V[0] ^ (M*N) ^ Z2[0]  # 第一个像素点的加密
        for i in range(1, M*N):
            V[i] = V[i] ^ V[i - 1] ^ Z2[i]
        E = np.rot90(V.reshape(M, N).T, -2)  # 获得加密图像
        E_BGR.append(E)
    E_BGR = E_BGR[::-1]
    E = cv2.merge(E_BGR)
    cv2.imwrite(encrypt_img_path, E)  # 保存图像

    # 显示加密图像
    if show:
        print('加密完成!')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
        plt.imshow(E, cmap='gray')
        plt.title('加密图像')
        plt.show()

    # 保存参数
    # 01列表压缩为 int 整数
    if not use_params:
        A11_0_int = int(
            '1' + ''.join(map(lambda x: str(reduce(lambda m, n: str(m)+str(n), x)), A11_0)), 2)
        A22_0_int = int(
            '1' + ''.join(map(lambda x: str(reduce(lambda m, n: str(m)+str(n), x)), A22_0)), 2)
        if not os.path.exists('./params'):
            os.mkdir('./params')
        np.savez(f'./params/{params_path}', x0=x0, u1=u1, y0=y0,
                 u2=u2, A11_0=A11_0_int, A22_0=A22_0_int, n_round=n_round)
    return encrypt_img_path, E


if __name__ == '__main__':
    img_path = "./data/rabbit1.jpg"  # 图像路径
    params = {
        'x0': None,
        'u1': None,
        'y0': None,
        'u2': None,
        'n_round': 1  # 加密轮数
    }
    encrypt(img_path, **params)
