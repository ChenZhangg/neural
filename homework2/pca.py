from PIL import Image
import numpy as np
import math
from sklearn.decomposition import PCA

def loadImage():
    img = Image.open("/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/face/s2/1.bmp")
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    #print(img.format, img.size, img.mode)
    #print(list(data))
    # 为了避免溢出，这里对数据进行一个缩放，缩小100倍
    data = np.array(data).reshape(height, width)
    #print(data.dtype)
    #print(data.shape)
    #print(data)
    # 查看原图的话，需要还原数据
    new_im = Image.fromarray(data.astype(np.uint8))
    #new_im.show()
    return data

def pca():
    data = loadImage()
    pca = PCA(n_components=10).fit(data)
    # 降维
    x_new = pca.transform(data)
    # 还原降维后的数据到原空间
    recdata = pca.inverse_transform(x_new)
    # 还原降维后的数据
    newImg = Image.fromarray(recdata)
    return recdata

def myPCA(k = 10):
    data = loadImage()
    n_features, n_samples = data.shape
    #print(n_samples)
    # 求均值
    mean = np.array([np.mean(data[i, :]) for i in range(n_features)])
    #print(mean)
    #print(mean.shape)
    mean = np.tile(mean.reshape(n_features, 1), (1, n_samples))
    # 去中心化
    normal_data = data - mean
    #print(data)
    #print(normal_data)
    #print(normal_data.shape)
    #matrix_ = np.dot(normal_data, np.transpose(normal_data))
    matrix_ = np.cov(normal_data)
    #print(matrix_.shape)
    eig_val, eig_vec = np.linalg.eig(matrix_)
    #print(eig_val)
    #print(eig_vec.shape)
    eigIndex = np.argsort(eig_val)
    #print(eigIndex)
    eigVecIndex = eigIndex[:-(k+1):-1]
    #print(eigVecIndex)
    feature = eig_vec[:,eigVecIndex]
    #print(feature.shape)
    print(feature)
    new_data = np.dot(feature.transpose(), normal_data)
    #print(new_data.shape)
    #print(feature.shape)
    # 将降维后的数据映射回原空间
    rec_data = np.dot(feature, new_data) + mean
    # print(rec_data)
    # 压缩后的数据也需要乘100还原成RGB值的范围
    print(rec_data.dtype)
    newImage = Image.fromarray(rec_data.astype(np.uint8))
    newImage.show()
    return rec_data

def psnr():
    data = loadImage()
    rec_data = myPCA(k=20)
    #rec_data = pca()
    mse = np.mean((data - rec_data) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


#myPCA()
print(psnr())