import cv2
from matplotlib import pyplot as plt

# 生成直方图


def histogram(img_path):
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.hist(src.ravel(), 256)
    plt.show()


if __name__ == "__main__":
    img_path = "../images/lena_encrypt.png"
    histogram(img_path)
