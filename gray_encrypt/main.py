from analysis import *
from encrypt import encrypt
from decrypt import decrypt
img_path = "./images/lena.png"
bifurcation_diagram(0.2, 100, 10)  # 分岔图
# encrypt_img_path, res = encrypt(img_path, show=True)  # 加密
# decrypt(encrypt_img_path, img_path, show=True)  # 解密
# histogram(encrypt_img_path)  # 直方图
# correlation(img_path, encrypt_img_path)  # 相关性
# entropy(img_path, encrypt_img_path)  # 信息熵
# encrypt_sensitivity(img_path)  # 加密敏感度
# decrypt_sensitivity(img_path, encrypt_img_path)  # 解密敏感度
# differencial(img_path)  # 差分攻击，NPCR和UACI
