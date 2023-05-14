import cv2
import matplotlib.pyplot as plt

'''
绘制灰度直方图
'''


def hist(img, is_raw=True):
    img = cv2.imread(img)
    B, G, R = cv2.split(img)
    # 转成一维
    R = R.flatten(order='C')
    G = G.flatten(order='C')
    B = B.flatten(order='C')

    # 结果展示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(221)
    plt.hist(img.flatten(order='C'), bins=range(257), color='gray')
    plt.title('原始图像' if is_raw else '加密图像')

    # 子图2，通道R
    plt.subplot(222)
    # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.hist(R, bins=range(257), color='red')
    plt.title('通道R')

    # 子图3，通道G
    plt.subplot(223)
    plt.hist(G, bins=range(257), color='green')
    plt.title('通道G')

    # 子图4，通道B
    plt.subplot(224)
    plt.hist(B, bins=range(257), color='blue')
    plt.title('通道B')
    # #设置子图默认的间距
    plt.tight_layout()
    plt.show()


def histogram(raw_img_path, encrypt_img_path):
    hist(raw_img_path)
    hist(encrypt_img_path, is_raw=False)


if __name__ == '__main__':
    img_path = "../images/lena_encrypt.png"
    histogram(img_path)
