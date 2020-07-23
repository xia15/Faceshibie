import numpy
from numpy import *
import math
import os
import cv2
import scipy.misc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
class Softmax(object):

    def __init__(self):
        self.learning_step = 0.000001           # 学习速率
        self.max_iteration = 100000             # 最大迭代次数
        self.weight_lambda = 0.01               # 衰退权重

    def cal_e(self,x,l):
        '''
        计算博客中的公式3
        '''

        theta_l = self.w[l]
        product = np.dot(theta_l,x)

        return math.exp(product)

    def cal_probability(self,x,j):
        '''
        计算博客中的公式2
        '''

        molecule = self.cal_e(x,j)
        denominator = sum([self.cal_e(x,i) for i in range(self.k)])

        return molecule/denominator


    def cal_partial_derivative(self,x,y,j):
        '''
        计算博客中的公式1
        '''

        first = int(y==j)                           # 计算示性函数
        second = self.cal_probability(x,j)          # 计算后面那个概率

        return -x*(first-second) + self.weight_lambda*self.w[j]

    def predict_(self, x):
        result = np.dot(self.w,x)
        row, column = result.shape

        # 找最大值所在的列
        _positon = np.argmax(result)
        m, n = divmod(_positon, column)

        return m

    def train(self, features, labels):
        self.k = len(set(labels))

        self.w = np.zeros((self.k,len(features[0])+1))
        time = 0

        while time < self.max_iteration:
            print('loop %d' % time)
            time += 1
            index = random.randint(0, len(labels) - 1)

            x = features[index]
            y = labels[index]

            x = list(x)
            x.append(1.0)
            x = np.array(x)

            derivatives = [self.cal_partial_derivative(x,y,j) for j in range(self.k)]

            for j in range(self.k):
                self.w[j] -= self.learning_step * derivatives[j]

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)

            x = np.matrix(x)
            x = np.transpose(x)

            labels.append(self.predict_(x))
        return labels


def LBP(src, n_points=8, radius=1, h=None, w=None):
    LBPoperator = mat(zeros(shape(src)))
    for i in range(shape(src)[0]):
        face = src[i, :].reshape(h, w)
        height, width = shape(face)
        dst = face.copy()
        src.astype(dtype=numpy.float32)
        dst.astype(dtype=numpy.float32)

        neighbours = numpy.zeros((1, n_points), dtype=numpy.uint8)
        lbp_value = numpy.zeros((1, n_points), dtype=numpy.uint8)
        for x in range(radius, width - radius - 1):
            for y in range(radius, height - radius - 1):
                lbp = 0
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
        LBPoperator[i, :] = dst.flatten()
    return LBPoperator

def calHistogram1(LBPoperator,hblock,wblock):
    exHistograms = mat(zeros(((shape(LBPoperator)[0]), 256*hblock*wblock)))
    # print(exHistograms.shape)
    for i in range(shape(LBPoperator)[0]):
        img = LBPoperator[i, :].reshape(150, 150)  # img= ImgLBPope[,j].reshape(125, 94)
        H, W = shape(img)
        # 把图片分为15*15份
        Histogram = mat(zeros((hblock*wblock, 256)))  # Histogram为256行乘以25列的全零矩阵
        maskx, masky = int(H / hblock), int(W / wblock)  # maskx = 20 ,masky = 20
        for k in range(hblock):
            for j in range(wblock):
                # 使用掩膜opencv来获得子矩阵直方图
                mask = zeros(shape(img), uint8)
                mask[int(k * maskx): int((k + 1) * maskx), int(j * masky):int((j + 1) * masky)] = 255
                # mask[0:29,0:14]=255  mask[0:29,14:28]=255
                hist = cv2.calcHist([array(img, uint8)], [0], mask, [256], [0, 255])
                hist = cv2.normalize(hist, None)
                Histogram[(k + 1) * (j + 1) - 1, :] = mat(hist).flatten()
        exHistograms[i, :] = Histogram.flatten()
    return exHistograms

def _load_image(file_paths, slice_, color, resize):
    default_slice = (slice(0, 150), slice(0, 150))
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
        print(h)
        w = int(resize * w)
        print(w)

    n_faces = len(file_paths)
    if not color:
        faces = numpy.zeros((n_faces, h, w), dtype=numpy.float32)
    else:
        faces = numpy.zeros((n_faces, h, w, 3), dtype=numpy.float32)

    for i , file_path in enumerate(file_paths):
        img = cv2.imread(file_path, 0)

        print(img.shape)
        face = numpy.asarray(img[slice_], dtype=numpy.float32)
        print(face.shape)
        # face /= 255.0
        if resize is not None:
             face = scipy.misc.imresize(face, resize)
        faces[i, ...] = face

    return faces

def _load_paths(data_folder_path, slice_=None, color=False, min_faces_per_person=0, resize=None):
    person_names, file_paths = [], []
    for person_name in sorted(os.listdir(data_folder_path)):
       folder_path = os.path.join(data_folder_path, person_name)
       if not os.path.isdir(folder_path):
           continue
       paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))]
       n_pictures = len(paths)
       if n_pictures >= min_faces_per_person:
           person_name = person_name.replace('_', ' ')
           person_names.extend([person_name] * n_pictures)
           file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError("min_faces_per_person=%d is too restrictive" %min_face_per_person)
    target_names = numpy.unique(person_names)
    target = numpy.searchsorted(target_names, person_names)
    faces = _load_image(file_paths, slice_, color, resize)

    indices = numpy.arange(n_faces)
    numpy.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]

    return faces, target, target_names

faces, target, target_names = _load_paths("pictureTest/", slice_=None, color=False,
                                          min_faces_per_person=0, resize=None)
rowsum, imgheight , imgwidth =shape(faces)
print("图片高为{0}".format(imgheight))
print("图片高为{0}".format(imgwidth))
X = faces.reshape(len(faces), -1)
print(X.shape)
X = mat(X)
#获取图片的LBP特征对象
LBPoperator = LBP(X,h=imgheight, w=imgwidth)
#把图片分成15*15个小块
exHistogram = calHistogram1(LBPoperator,15,15)

print(exHistogram)
#print(type(exHistogram.A))
exHistogram = exHistogram.A
exHistogram = exHistogram.flatten()
#print(exHistogram.shape)


LBPoperator = LBPoperator.reshape(imgheight, imgwidth)
print(type(LBPoperator))
print(LBPoperator.shape)
plt.imshow(LBPoperator, plt.cm.gray)
plt.savefig("LBP人脸特征图.png")
plt.show()


arr = LBPoperator.flatten().T
print(arr.shape)
plt.hist(arr, bins=256, facecolor='blue')
plt.savefig("LBP直方图.png")
plt.show()
n_components = 80
pca = decomposition.PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(LBPoperator)
X_train_pca = pca.transform(LBPoperator)
print(X_train_pca.shape)
plt.subplot(111)
arr = X_train_pca.flatten()
print(arr.shape)
plt.hist(arr, bins=256, facecolor='blue')
plt.savefig("LBP降维后直方图")
plt.show()
