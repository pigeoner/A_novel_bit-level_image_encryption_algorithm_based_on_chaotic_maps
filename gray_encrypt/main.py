from analysis import *
from encrypt import encrypt
from decrypt import decrypt
img_path = "./images/rabbit.jpg"
encrypt_img_path, res = encrypt(img_path, show=True)
decrypt(encrypt_img_path, img_path, show=True)
histogram(encrypt_img_path)
correlation(img_path, encrypt_img_path)
entropy(img_path, encrypt_img_path)
encrypt_sensitivity(img_path)
decrypt_sensitivity(img_path, encrypt_img_path)
differencial(img_path)
