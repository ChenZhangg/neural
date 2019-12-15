"""
File: RBFN.py
Author: Octavio Arriaga
Email: arriaga.camargo@email.com
Github: https://github.com/oarriaga
Description: Minimal implementation of a radial basis function network
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import exp
import os
from PIL import Image
from sklearn.decomposition import PCA
#高斯核函数
def rbf(x, c, s):
    """
    核函数
    :param x: 输入单个数据
    :param c: 中心值
    :param s: 偏差
    :return:
    """
    return exp(-1 / (2 * s**2) * (x-c)**2)


class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(np.array(center) - np.array(data_point))**2)

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        # 随机选取隐层神经元个数个中心点
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = []
        for i in random_args:
            centers.append(X[i])
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions


def dimension_reduction(data, k):
    pca = PCA(n_components=k).fit_transform(data)
    return pca

def load_data(k, ratio = 5):
    ary = None
    for i in (3, 34):
        dir_path = "/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/face/s{}/".format(i + 1)
        files = os.listdir(dir_path)
        for file_name in files:
            if file_name.endswith("bmp"):
                file_path = dir_path + file_name
                #print(file_path)
                img = Image.open(file_path)
                data = list(img.getdata())
                data = np.array(data).reshape(-1, 1) / 255
                if ary is None:
                    ary = data
                else:
                    ary = np.hstack((ary, data))

    feature = dimension_reduction(ary, k)
    data = np.dot(feature.transpose(), ary)

    train_X = []
    train_y = []
    test_X = []
    test_y = []
    #50%测试数据
    for i in (list(range(0, ratio)) + list(range(10, ratio + 10))):
        train_X.append(data[:, i].transpose().tolist())
        if(i < 10):
            train_y.append(0)
        else:
            train_y.append(1)
    for i in (list(range(ratio, 10)) + list(range(ratio + 10, 20))):
        test_X.append(data[:, i].tolist())
        if (i < 10):
            test_y.append(0)
        else:
            test_y.append(1)
    return train_X, train_y, test_X, test_y

def run():
    train_X, train_y, test_X, test_y = load_data(20, 8)
    model = RBFN(hidden_shape=5, sigma=1.)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    print(y_pred)

run()



