import math
import cv2
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import face_recognition
from collections import defaultdict
import numpy as np
import os

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    else:
        return 0


if __name__ == '__main__':
    li= os.listdir("./lfw_align/")
    mkdir('./lfw_gray')
    print('%d'%len(li))
    # 遍历文件夹,获取文件夹名
    for i in range(0, len(li)):
        fileName = os.path.basename(li[i])
        p="./lfw_gray/"+fileName
        mkdir(p)
        paths = "./lfw_align/" + fileName
        filenames = os.listdir(paths)
        filenames.sort()
        # 遍历每个文件下的文件，获取文件名
        for filename in filenames:
            out_path = paths +"/"+ filename
            # #读入图片
            img = cv2.imread(out_path)
            #转换成灰度图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GARY)
            origin_img = Image.fromarray(img)
            path = "./lfw_align/" + fileName + "/" + filename
            origin_img.save(path)
