from sklearn.decomposition import FastICA
from PIL import Image
import numpy as np

img1 = Image.open("/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/风景图/1.bmp")
img2 = Image.open("/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/风景图/2.bmp")
img3 = Image.open("/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/风景图/3.bmp")
img4 = Image.open("/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/风景图/4.bmp")
width = img1.size[0]
height = img1.size[1]
data1 = np.array(img1.getdata()).reshape(width * height, 1) / 255
data2 = np.array(img2.getdata()).reshape(width * height, 1) / 255
data3 = np.array(img3.getdata()).reshape(width * height, 1) / 255
data4 = np.array(img4.getdata()).reshape(width * height, 1) / 255
print(data1)
data = np.hstack((data1, data2, data3, data4))
mix = np.random.rand(4, 4)
X = np.dot(data, mix)
# blend1 = data1 * 0.1 + data2 * 0.2 + data3 * 0.3 + data4 * 0.4
# blend2 = data1 * 0.4 + data2 * 0.1 + data3 * 0.2 + data4 * 0.3
# blend3 = data1 * 0.3 + data2 * 0.4 + data3 * 0.1 + data4 * 0.2
# blend4 = data1 * 0.2 + data2 * 0.3 + data3 * 0.4 + data4 * 0.1
print(X.shape)
transformer = FastICA()
X_transformed = transformer.fit_transform(X)
print(X_transformed.shape)
for i in range(1):
    tmp = X_transformed[:, i].reshape(height, width)
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 255
    print(tmp)
    new_im = Image.fromarray(tmp.astype(np.uint8))
    new_im.show()