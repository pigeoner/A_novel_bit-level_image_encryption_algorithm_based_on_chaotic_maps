from analysis import *
from encrypt import encrypt
from decrypt import decrypt
import os
img_path_base = "./images/"

img_name = "lena.png"  # 自行修改图片名，默认为 lena 图 

img_path = img_path_base + img_name

if os.path.exists('result.txt'):
    os.remove('result.txt')
if not os.path.exists('params/'):
    os.mkdir('params/')

# 加解密过程及性能分析
with open('result.txt','w', encoding='utf8') as f:
    f.write(
""" +---------------------------------+
|        加解密过程的性能分析         |
 +---------------------------------+

""")
bifurcation_diagram(0.2, 100, 10)  # 分岔图
encrypt_img_path, res = encrypt(img_path, show=True)  # 加密
decrypt(encrypt_img_path, img_path, show=True)
histogram(img_path, encrypt_img_path)  # 直方图
correlation(img_path, encrypt_img_path)  # 相关性
entropy(img_path, encrypt_img_path)  # 信息熵
encrypt_sensitivity(img_path)  # 加密敏感度
decrypt_sensitivity(img_path, encrypt_img_path)  # 解密敏感度
differencial(img_path)  # 差分攻击，NPCR和UACI

# 100次相关性分析和差分攻击分析
with open('result.txt','a+', encoding='utf8') as f:
    f.write("""

 +---------------------------------+
|    100次相关性分析和差分攻击分析    |
 +---------------------------------+

""")
correlation(img_path, encrypt_img_path, is_mean=True, ntimes=100) # 100次相关性分析结果平均值
differencial(img_path, ntimes=100) # 100次差分攻击结果平均值

print('程序执行完毕，结果已保存至result.txt')