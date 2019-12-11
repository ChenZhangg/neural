import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import IncrementalPCA

class CCIPCA(object):
    def __init__(self, inputDim, outputDim):
        self._input_dim = inputDim
        self._output_dim = outputDim
        self._n = 1
        self._mean = 0.1 * np.random.randn(1, self._input_dim)
        #self._mean = np.zeros((1, self._input_dim))
        self._eigenVectors = 0.1 * np.random.randn(self._output_dim, self._input_dim)
        #self._eigenVectors = np.zeros((self._output_dim, self._input_dim))

    def _amnestic(self, t):  # amnestic function
        [n1, n2, a, C] = [20., 200., 2000., 2.]
        if t < n1:
            alpha = 0
        elif (t >= n1) and (t < n2):
            alpha = C * (t - n1) / (n2 - n1)
        else:
            alpha = C + (t - n2) / a

        n = t
        _lr = float(n - alpha - 1) / n  # learning rate
        _rr = float(alpha + 1) / n  # residual rate

        return [_rr, _lr]

    def update(self, x):
        assert (x.shape[0] == 1)

        # compute the mean imcrementally
        self._mean = float(self._n - 1) / self._n * self._mean + float(1) / self._n * x

        if self._n > 1:
            u = x - self._mean  # reduce the mean vector
            [w1, w2] = self._amnestic(self._n)  # conpute the amnestic parameters
            k = min(self._n, self._output_dim)
            for i in range(k):  # update all eigenVectors
                v = self._eigenVectors[i, :].copy()  # get the current eigenVector
                if (i == k - 1):
                    v = u.copy()
                    vn = v / np.linalg.norm(v)  # normalize the vector
                else:
                    v = w1 * v + w2 * np.dot(u, v.T) / np.linalg.norm(v) * u  # update the eigenVector
                    vn = v / np.linalg.norm(v)  # normalize the vector

                u = u - np.dot(u, vn.T) * vn  # remove the projection of u on the v
                self._eigenVectors[i, :] = v.copy()

        self._n += 1  # update the mean of the data
"""
    def update(self, x):
        assert (x.shape[0] == 1)
        #print(x.shape)
        # compute the mean imcrementally
        # 输入样本求均值
        self._mean = float(self._n - 1) / self._n * self._mean + float(1) / self._n * x
        if self._n == 1:
            self._eigenVectors[0, :] = x

        if self._n > 1:
            u = x - self._mean  # reduce the mean vector 去均值
            #[w1, w2] = self._amnestic(self._n)  # compute the amnestic parameters
            k = min(self._n, self._output_dim)
            for i in range(k):  # update all eigenVectors
                v = self._eigenVectors[i, :].copy()  # get the current eigenVector
                if (i == k - 1):
                    v = u.copy()
                    vn = v / np.linalg.norm(v)  # normalize the vector
                else:
                    v = float(self._n - 1) / self._n * v + float(1) / self._n * np.dot(u, u.T) / np.linalg.norm(v) * v  # update the eigenVector
                    vn = v / np.linalg.norm(v)  # normalize the vector

                u = u - np.dot(u, vn.T) * vn  # remove the projection of u on the v
                #print(v)
                self._eigenVectors[i, :] = v.copy()

        print("self._eigenVectors" + str(self._n))
        print(self._eigenVectors)
        self._n += 1  # update the mean of the data
"""

#"""
#设置主元个数
k = 10
filename = "/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/face/s2/1.bmp"
img = Image.open(filename)
width = img.size[0]
height = img.size[1]
#print(height)
#print(width)
ccipca = CCIPCA(height, k)
data = img.getdata()
data = np.array(data).reshape(height, width)
#mean = np.array([np.mean(data[i, :]) for i in range(height)])
#mean = np.tile(mean.reshape(height, 1), (1, width))

for i in range(width):
    im = data[:, i].reshape(1, -1)
    #print(im)
    ccipca.update(im)
print("ccipca._mean.shape")
print(ccipca._mean.shape)
mean = np.tile(ccipca._mean.copy().reshape(height, 1), (1, width))
normal_data = data - mean
print("normal_data")
print(normal_data)
print("ccipca._eigenVectors")
#print(ccipca._eigenVectors.shape)
print(ccipca._eigenVectors)
print(ccipca._eigenVectors.shape[0])
eigenVectors = ccipca._eigenVectors
feature = np.zeros(eigenVectors.shape)
for i in range(eigenVectors.shape[0]):
    mod = np.linalg.norm(eigenVectors[i, :])
    feature[i, :] = eigenVectors[i, :].copy() / mod
new_data = np.dot(feature, normal_data)

print("new_data")
print(new_data)
#print(new_data.shape)
#print(feature.shape)
# 将降维后的数据映射回原空间
rec_data = np.dot(feature.transpose(), new_data) + mean
print("rec_data")
print(rec_data)
print("data")
print(data)
# 压缩后的数据也需要乘100还原成RGB值的范围
newImage = Image.fromarray(rec_data.astype(np.uint8))
#newImage = Image.fromarray(data.astype(np.uint8))
print(ccipca._mean.shape)
newImage.show()
#"""

"""
filename = "/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/face/s2/1.bmp"
img = Image.open(filename)
width = img.size[0]
height = img.size[1]
data = img.getdata()
data = np.array(data).reshape(height, width)
transformer = IncrementalPCA(n_components=7)
x_new = transformer.fit_transform(data)
# 还原降维后的数据到原空间
recdata = transformer.inverse_transform(x_new)
newImg = Image.fromarray(recdata)
newImg.show()
"""

"""
#设置主元个数
k = 5
filename = "/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/face/s2/1.bmp"
img = Image.open(filename)
width = img.size[0]
height = img.size[1]
data = img.getdata()
data = np.array(data).reshape(height, width)
w=np.zeros((height, width))
xmean = data[:, 0]
print(xmean.shape)
w[:, 0] = data[:, 0]
for t in range(1, width):
    xt = data[:, t]
    xmean = float(t) / (t + 1) * xmean + float(1) / (t + 1) * xt
    xt = xt - xmean
    m = min(t, width)
    for i in range(m):
        if(i == (t - 1)):
            w[:, i] = xt
        else:
            w[:, i] = (t / (t + 1)) * w[:, i] + (1 / (t + 1)) * np.dot(xt, xt.T) / np.linalg.norm(w[:, i])
            xt = xt - xt.T * np.linalg.norm(w[:, i]) * np.linalg.norm(w[:, i])
print(w[:, 0:k])
print(w.shape)
"""