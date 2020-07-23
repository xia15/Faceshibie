import numpy
from numpy import *
import math
import os
import cv2
import scipy.misc
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.decomposition as decomposition
import matplotlib.pyplot as plt
import time

def _load_image(file_paths, slice_, color, resize):
    #照片的长宽为504*480
    #default_slice = (slice(0, 480), slice(0, 504))
    # -----------------------------------------------------------------------
    #识别lfw数据集
    default_slice = (slice(0, 149), slice(0, 149))
    # -----------------------------------------------------------------------
    # #识别orl数据集
    # default_slice = (slice(0, 92), slice(0, 92))
    # -----------------------------------------------------------------------
    # #识别yale数据集
    # default_slice = (slice(0, 100), slice(0, 100))
    # -----------------------------------------------------------------------
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    n_faces = len(file_paths)
    if not color:
        faces = numpy.zeros((n_faces, h, w), dtype=numpy.float32)
    else:
        faces = numpy.zeros((n_faces, h, w, 3), dtype=numpy.float32)

    for i, file_path in enumerate(file_paths):
        img = cv2.imread(file_path, 0)
        face = numpy.asarray(img[slice_], dtype=numpy.float32)
        if resize is not None:
             face = scipy.misc.imresize(face, resize)
        faces[i, ...] = face

    return faces

#读取文件夹下的图片，返回图片列表
def _load_paths(data_folder_path, slice_=None, color=False, min_faces_per_person=0, resize=None):
    # 人名集合和人脸图片路径
    person_names, file_paths = [], []
    #读取人员姓名文件夹，返回文件名列表
    for person_name in sorted(os.listdir(data_folder_path)):
        #文件路径拼接
       folder_path = os.path.join(data_folder_path, person_name)
       #判断folder_path是否为一个目录
       if not os.path.isdir(folder_path):
           continue
        #将人名文件夹和文件夹下的人名文件拼接,paths为一个人名文件夹下图片完整路径列表
       paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))]
       n_pictures = len(paths)
       #如果一个人名文件夹出现多个人名，对人名集合进行扩展
       if n_pictures >= min_faces_per_person:
           person_name = person_name.replace('_', ' ')
           # 这儿用extend是因为extend可以一次性添加多个元素，append一次只能添加一个，扩展列表
           person_names.extend([person_name] * n_pictures)
           file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError("min_faces_per_person=%d is too restrictive" %min_face_per_person)
    # 通过unique函数得到所有不同人名
    target_names = numpy.unique(person_names)
    # searchsorted是寻求插入位置的函数，在这儿巧妙的将person_names数字化，
    # target代表person_names中每个名字在target_names的位置
    target = numpy.searchsorted(target_names, person_names)
    # 加载人脸，slice_是对人脸切片，lolor用来指定加载彩色还是黑白图片，resize为缩放比例
    faces = _load_image(file_paths, slice_, color, resize)

    #使用确定性RNG方案洗牌，以避免同一个人的所有面孔排成一行，因为这会破坏一些交叉验证和学习算法，如SGD和online
    #k-means是IID假设
    # 这儿就是打乱人脸顺序的工作了
    indices = numpy.arange(n_faces)
    numpy.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]

    return faces, target, target_names

#LBP特征提取
def LBP(src, n_points=8, radius=2, h=None, w=None):
    LBPoperator = mat(zeros(shape(src)))
    for i in range(shape(src)[0]):
        print("开始处理第{0}图片".format(i))
        face = src[i, :].reshape(h, w)
        height, width = shape(face)
        dst = face.copy()
        src.astype(dtype=numpy.float32)
        dst.astype(dtype=numpy.float32)

        neighbours = numpy.zeros((1, n_points), dtype=numpy.uint8)
        lbp_value = numpy.zeros((1, n_points), dtype=numpy.uint8)
        for x in range(radius, width - radius - 1):
            for y in range(radius, height - radius - 1):
                lbp = 0.
                # 先计算共n_points个点对应的像素值，使用双线性插值法
                for n in range(n_points):
                    theta = float(2 * numpy.pi * n) / n_points
                    x_n = x + radius * numpy.cos(theta)
                    y_n = y - radius * numpy.sin(theta)

                    # 向下取整
                    x1 = int(math.floor(x_n))
                    y1 = int(math.floor(y_n))
                    # 向上取整
                    x2 = int(math.ceil(x_n))
                    y2 = int(math.ceil(y_n))

                    # 将坐标映射到0-1之间
                    tx = numpy.abs(x - x1)
                    ty = numpy.abs(y - y1)

                    # 根据0-1之间的x，y的权重计算公式计算权重
                    w1 = (1 - tx) * (1 - ty)
                    w2 = tx * (1 - ty)
                    w3 = (1 - tx) * ty
                    w4 = tx * ty

                    # 根据双线性插值公式计算第k个采样点的灰度值
                    neighbour = face[y1, x1] * w1 + face[y2, x1] * w2 + face[y1, x2] * w3 + face[y2, x2] * w4

                    neighbours[0, n] = neighbour

                center = face[y, x]

                for n in range(n_points):
                    if neighbours[0, n] > center:
                        lbp_value[0, n] = 1
                    else:
                        lbp_value[0, n] = 0

                for n in range(n_points):
                    lbp += lbp_value[0, n] * 2**n

                # 转换到0-255的灰度空间，比如n_points=16位时结果会超出这个范围，对该结果归一化
                lbp = int(lbp / (2**n_points-1) * 255)
                dst[y, x] = lbp
                # #保持lbp旋转不变性
                #minValue = minBinary(lbp)
                # # dst[y, x] =minValue
                # #查找lbp等价模式表
                #value58 = BuildUniformPatternTable(minValue)
                # dst[y, x] = value58
                # #计算9种等价模式
                #NineofModel = ComputeValue9(value58)
                #dst[y, x] = NineofModel
        LBPoperator[i, :] = dst.flatten()
    return LBPoperator

#统计直方图
def calHistogram1(LBPoperator,hblock,wblock):
    exHistograms = mat(zeros(((shape(LBPoperator)[0]), 256*hblock*wblock)))
    # print(exHistograms.shape)
    for i in range(shape(LBPoperator)[0]):
        #288对应480，302对应504*0.6
        #img = LBPoperator[i, :].reshape(288, 302)
        # -----------------------------------------------------------------------
        #lfw数据集
        img = LBPoperator[i, :].reshape(89, 89)
        # -----------------------------------------------------------------------
        # # orl数据集
        # img = LBPoperator[i, :].reshape(55, 55)
        # -----------------------------------------------------------------------
        # # yale数据集
        # img = LBPoperator[i, :].reshape(60, 60)
        # -----------------------------------------------------------------------
        H, W = shape(img)
        # 把图片分为15*15份
        Histogram = mat(zeros((hblock*wblock, 256)))  # Histogram为256行乘以15*15列的全零矩阵
        maskx, masky = int(H / hblock), int(W / wblock)
        for k in range(hblock):
            for j in range(wblock):
                # 使用掩膜opencv来获得子矩阵直方图
                mask = zeros(shape(img), uint8)
                mask[int(k * maskx): int((k + 1) * maskx), int(j * masky):int((j + 1) * masky)] = 255
                # mask[0:29,0:14]=255  mask[0:29,14:28]=255
                hist = cv2.calcHist([array(img, uint8)], [0], mask, [256], [0, 255])
                #归一化处理
                hist = cv2.normalize(hist, None)
                Histogram[(k + 1) * (j + 1) - 1, :] = mat(hist).flatten()
        exHistograms[i, :] = Histogram.flatten()
    return exHistograms


#载入图片
# faces, target, target_names = _load_paths("yale_faces/", slice_=None, color=False,
#                                           min_faces_per_person=1, resize=0.6)
# # #
# rowsum, imgheight , imgwidth =shape(faces)
# X = faces.reshape(len(faces), -1)
# X = mat(X)
# y = target

# # # 训练集
#-----------------------------------------------------------------------
#lfw数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# numpy.savetxt("target_nameslfwPlus.txt", target_names, fmt='%s')
# fingerLBP = LBP(X_train, radius=1, h=imgheight, w=imgwidth)  # 获得实验图像LBP算子，每一行保存一张图片的LBP信息
# numpy.savetxt("lfwLBPtrainPlus.txt", fingerLBP, fmt="%d")
# numpy.savetxt("y_trainlfwPlus.txt", y_train, fmt="%d")
#
# # # # 测试集
#
# fingerLBPtest = LBP(X_test, radius=1, h=imgheight, w=imgwidth)  # 获得实验图像LBP算子，例如
# numpy.savetxt("lfwLBPtestPlus.txt", fingerLBPtest, fmt="%d")
# numpy.savetxt("y_testlfwPlus.txt", y_test, fmt="%d")
# #载入数据集
fingerLBPtrain = numpy.loadtxt("lfwLBPtrainPlus.txt")
# print(fingerLBPtrain.shape)
y_train = numpy.loadtxt("y_trainlfwPlus.txt")
fingerLBPtest = numpy.loadtxt("lfwLBPtestPlus.txt")
y_test = numpy.loadtxt("y_testlfwPlus.txt")

#-----------------------------------------------------------------------
#orl数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# numpy.savetxt("target_namesORLPlus.txt", target_names, fmt='%s')
# fingerLBP = LBP(X_train, radius=1, h=imgheight, w=imgwidth)  # 获得实验图像LBP算子，每一行保存一张图片的LBP信息
# numpy.savetxt("ORLLBPtrainPlus.txt", fingerLBP, fmt="%d")
# numpy.savetxt("y_trainORLPlus.txt", y_train, fmt="%d")
#
# # # # 测试集
#
# fingerLBPtest = LBP(X_test, radius=1, h=imgheight, w=imgwidth)  # 获得实验图像LBP算子，例如
# numpy.savetxt("ORLLBPtestPlus.txt", fingerLBPtest, fmt="%d")
# numpy.savetxt("y_testORLPlus.txt", y_test, fmt="%d")
# #载入数据集
# fingerLBPtrain = numpy.loadtxt("ORLLBPtrainPlus.txt")
# # print(fingerLBPtrain.shape)
# y_train = numpy.loadtxt("y_trainORLPlus.txt")
# fingerLBPtest = numpy.loadtxt("ORLLBPtestPlus.txt")
# y_test = numpy.loadtxt("y_testORLPlus.txt")

#按比例划分测试机和训练集，固定随机数种子(42)，这样每次得到的训练集数据相同
#-----------------------------------------------------------------------
# yale数据集
#训练集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#
# numpy.savetxt("target_namesyalePlus.txt", target_names, fmt='%s')
# fingerLBP = LBP(X_train, radius=1, h=imgheight, w=imgwidth)  # 获得实验图像LBP算子，每一行保存一张图片的LBP信息
# numpy.savetxt("yaleLBPtrainPlus.txt", fingerLBP, fmt="%d")
# numpy.savetxt("y_trainyalePlus.txt", y_train, fmt="%d")
#
# # # # 测试集
#
# fingerLBPtest = LBP(X_test, radius=1, h=imgheight, w=imgwidth)  # 获得实验图像LBP算子，例如
# numpy.savetxt("yaleLBPtestPlus.txt", fingerLBPtest, fmt="%d")
# numpy.savetxt("y_testyalePlus.txt", y_test, fmt="%d")

# #载入生成的LBP文件
# fingerLBPtrain = numpy.loadtxt("yaleLBPtrainPlus.txt")
# # print(fingerLBPtrain.shape)
# y_train = numpy.loadtxt("y_trainyalePlus.txt")
# fingerLBPtest = numpy.loadtxt("yaleLBPtestPlus.txt")
# y_test = numpy.loadtxt("y_testyalePlus.txt")
#-----------------------------------------------------------------------

hblock = int(input("请输入想分的高的直方图块数："))
wblock = int(input("请输入想分的宽的直方图块数："))
exHistogram = calHistogram1(fingerLBPtrain, hblock, wblock)
exHistogram1 = calHistogram1(fingerLBPtest, hblock, wblock)


# 特征匹配
# 1、没有经过PCA降维的特征匹配
#没有使用PCA降维的特征匹配
    # hblock = int(input("请输入想分的高的直方图块数："))
    # wblock = int(input("请输入想分的宽的直方图块数："))
    # exHistogram = calHistogram1(fingerLBPtrain, hblock, wblock)
    # exHistogram1 = calHistogram1(fingerLBPtest, hblock, wblock)
LBPtimestart = time.time()
classifier = SVC(kernel="poly", gamma=2, C=5)
classifier.fit(exHistogram, y_train)
print(classifier.score(exHistogram1, y_test.ravel()))
LBPtimeend = time.time()
print("没有降维分类所用时间：{0}".format(LBPtimeend - LBPtimestart))

#2、使用PCA降维后的特征匹配
# for i in range(10, 107, 5):
#     PCAtimestart = time.time()
#     print("特征维数：{0}".format(i))
#     pca = decomposition.PCA(n_components=i, svd_solver="randomized", whiten=True).fit(exHistogram)
#     X_train_pca = pca.transform(exHistogram)
#     X_test_pca = pca.transform(exHistogram1)
#     classifier1 = SVC(kernel="rbf", gamma=0.01, C=10)
#     classifier1.fit(X_train_pca, y_train)
# # # y_pred1 = classifier1.predict(exHistogram1)
# # y_pred2 = classifier1.predict(X_test_pca)
#     print(classifier1.score(X_test_pca, y_test.ravel()))
#     PCAtimeend = time.time()
#     print("PCA降维分类所用时间：{0}".format((PCAtimeend - PCAtimestart)))
# #