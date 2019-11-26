from PIL import Image
import numpy as np

def myICA():
    img1 = Image.open("/Users/zhangchen/Documents/资料/学校材料/课程/神经网络及应用/NN作业用到的材料/风景图/1.bmp")
    img2 = Image.open("/Users/zhangchen/Documents/资料/学校材料/课程/神经网络及应用/NN作业用到的材料/风景图/2.bmp")
    img3 = Image.open("/Users/zhangchen/Documents/资料/学校材料/课程/神经网络及应用/NN作业用到的材料/风景图/3.bmp")
    img4 = Image.open("/Users/zhangchen/Documents/资料/学校材料/课程/神经网络及应用/NN作业用到的材料/风景图/4.bmp")
    width = img1.size[0]
    height = img1.size[1]
    data1 = np.array(img1.getdata()).reshape(1, width * height)
    data2 = np.array(img2.getdata()).reshape(1, width * height)
    data3 = np.array(img3.getdata()).reshape(1, width * height)
    data4 = np.array(img4.getdata()).reshape(1, width * height)
    blend1 = data1 * 0.1 + data2 * 0.2 + data3 * 0.3 + data4 * 0.4
    blend2 = data1 * 0.4 + data2 * 0.1 + data3 * 0.2 + data4 * 0.3
    blend3 = data1 * 0.3 + data2 * 0.4 + data3 * 0.1 + data4 * 0.2
    blend4 = data1 * 0.2 + data2 * 0.3 + data3 * 0.4 + data4 * 0.1
    blend = np.vstack((blend1, blend2, blend3, blend4))
    r, c = blend.shape
    mean = np.array([np.mean(blend[i, :]) for i in range(r)])
    for i in range(r):
        blend[i, :] = blend[i, :] - mean[i]  # 数据标准化，均值为零

    cx = np.cov(blend)
    value, eigvector = np.linalg.eig(cx)  # 计算协方差阵的特征值
    print(value)
    val = value**(-1/2) * np.eye(r, dtype = float)
    white = np.dot(val ,eigvector.T)  #白化矩阵
    z = np.dot(white, blend)
    z2 = np.dot(z, z.transpose())
    R, C = z2.shape
    z4 = np.dot(z2, z2.transpose()) / (4)
    print(z4.shape)
    print(z4)
    value, eigvector = np.linalg.eig(z4)  # 计算协方差阵的特征值
    print(z.shape)
    print(eigvector.shape)
    result = np.transpose(eigvector) @ z
    print(result.shape)
    print(mean)
    print(r)
    for i in range(r):
        result[i, :] = result[i, :] + 0  # 数据标准化，均值为零

    for i in  range(4):
        tmp = result[i, :].reshape(height, width)
        new_im = Image.fromarray(tmp.astype(np.uint8))
        new_im.show()

    #z4 =

myICA()