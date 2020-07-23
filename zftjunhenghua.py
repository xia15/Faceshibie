
# # #灰度图像直方图均衡化
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
image = cv2.imread('E:\\image\\00t.jpg')#读入灰度图像
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(img)#直方图均衡，返回值为均衡化后的图
plt.subplot(221),plt.imshow(image[:,:,[2,1,0]]),plt.title("原图")
plt.subplot(222),plt.imshow(equ,'gray'),plt.title("均衡化后的图")
plt.subplot(223),plt.hist(img.ravel(),256,[0,256]),plt.title("原图直方图")
plt.subplot(224),plt.hist(equ.ravel(),256,[0,256]),plt.title("均衡化后的直方图")
plt.show()


# # #彩图像直方图均衡化
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei']
img = cv2.imread('E:\\image\\a4.jpg')
(b,g,r)=cv2.split(img)#拆分图像
bH = cv2.equalizeHist(b)#合通道进行直方图均衡
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))#融合回原图像
plt.subplot(221),plt.imshow(img[:,:,[2,1,0]])
plt.title("原图")
plt.subplot(222),plt.imshow(result[:,:,[2,1,0]])
plt.title("均衡化后的图")
plt.subplot(223)
color = ('b', 'g', 'r')
for i , color in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color)
plt.title("原图直方图")
plt.subplot(224)
color = ('b', 'g', 'r')
for i , color in enumerate(color):
    hist = cv2.calcHist([result], [i], None, [256], [0, 256])
    plt.plot(hist, color)
plt.title("均衡化后的直方图")
plt.show()

# # #将其颜色空间转换至YUV空间，仅对其亮度空间进行直方图均衡
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei']
img = cv2.imread('E:\\image\\a3.jpg')
img_yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)#转换至YUV颜色空间
img_yuv[:,:,0]=cv2.equalizeHist(img_yuv[:,:,0])#对亮度空间进行直方图均衡
result = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)#转换回彩色图像
plt.subplot(221),plt.imshow(img[:,:,[2,1,0]]),plt.title("原图")
plt.subplot(222),plt.imshow(result[:,:,[2,1,0]]),plt.title("均衡化后的图")
plt.subplot(223)
color = ('b', 'g', 'r')
for i , color in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color)
plt.title("原图直方图")
plt.subplot(224)
color = ('b', 'g', 'r')
for i , color in enumerate(color):
    hist = cv2.calcHist([result], [i], None, [256], [0, 256])
    plt.plot(hist, color)
plt.title("均衡化后的直方图")
plt.show()
