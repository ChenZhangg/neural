import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image

class CCIPCA(object):
    def __init__(self, inputDim, outputDim):
        self._input_dim = inputDim
        self._output_dim = outputDim
        self._n = 1
        self._mean = 0.1 * np.random.randn(1, self._input_dim)
        self._eigenVectors = 0.1 * np.random.randn(self._output_dim, self._input_dim)

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
        #print(x.shape)
        # compute the mean imcrementally
        # 输入样本求均值
        self._mean = float(self._n - 1) / self._n * self._mean + float(1) / self._n * x

        if self._n > 1:
            u = x - self._mean  # reduce the mean vector
            [w1, w2] = self._amnestic(self._n)  # compute the amnestic parameters
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

#设置主元个数
k = 20
filename = "/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/face/s2/1.bmp"
img = Image.open(filename)
width = img.size[0]
height = img.size[1]
print(height)
print(width)
ccipca = CCIPCA(height, k)
data = img.getdata()
data = np.array(data).reshape(height, width)
for i in range(width):
    im = data[:, i].reshape(1, -1)
    #print(im)
    ccipca.update(im)

feature = None
for i in range(k):
    name = "eigenV" + str(i + 1)
    im = ccipca._eigenVectors[i, :].copy()
    im = np.array(im)
    im.reshape(1, -1)
    if feature is None:
        feature = im
    else:
        feature = np.vstack((feature, im))
    print(im.shape)
#print(feature.shape)
new_data = np.dot(feature, data)
#print(new_data.shape)
#print(feature.shape)
# 将降维后的数据映射回原空间
rec_data = np.dot(feature.transpose(), new_data) + ccipca._mean.reshape(-1, 1)
# print(rec_data)
# 压缩后的数据也需要乘100还原成RGB值的范围
newImage = Image.fromarray(rec_data.astype(np.uint8))
print(ccipca._mean.shape)
newImage.show()

"""
ccipca = CCIPCA(784, 20)
filename = 'train-images.idx3-ubyte'
#filename = "/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/face/s2/1.bmp"
binfile = open(filename, 'rb')
buf = binfile.read()

index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
print(magic)
print(numImages)
print(numRows)
print(numColumns)
index += struct.calcsize('>IIII')

for i in range(numImages):
    if (i % 10000 == 0):
        print("The current step of update is: ", i)
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    im = np.array(im, dtype=np.int8)
    im = im.reshape(1, 784)
    ccipca.update(im)

# ccipca.scaleTo255()
fig = plt.figure()
for i in range(20):
    name = "eigenV" + str(i + 1)
    im = ccipca._eigenVectors[i, :].copy()
    im = np.array(im)
    im = im.reshape(28, 28)
    plt.subplot(4, 5, i + 1), plt.title(name, fontsize=8)
    plt.imshow(im), plt.axis('off')
#     plt.tight_layout()

fig.tight_layout()  # 调整整体空白
plt.show()
"""