import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

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
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes = 10, random_state = 1, max_iter = 2000)
    clf.fit(train_X, train_y)
    for x in test_X:
        #print(x)
        print(clf.predict([x]))
run()