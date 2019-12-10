from math import exp
from math import sin
from math import pi
import numpy as np
import matplotlib.pyplot as plt

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


def kmeans(X, k):
    """k临近法处理一维数据

    Arguments:
        X {ndarray} -- Mx1 维输入
        k {int} -- 聚类数目

    Returns:
        ndarray -- A kx1 array of final cluster centers
    """

    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False

    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)

        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)

        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds

class RBFNet:
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=2, lr=0.005, epochs=2000, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        print(self.w)
        self.b = np.random.randn(1)
        print(self.b)

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
            #print(self.centers)
            #print(self.stds)
        else:
            # use a fixed std
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                # 计算核函数转换后的输入
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                #print(a)
                F = a.T.dot(self.w) + self.b
                #print("a.T" + str(a.T))
                #print("F" + str(F))
                loss = (y[i] - F).flatten() ** 2
                #print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)

n_sample = 1000

"""
X = np.linspace(-6 * pi, 6 * pi,n_sample)
y = []
for i in X:
    y.append(sin(i))
"""
"""
X = np.linspace(-6 * pi, 6 * pi, n_sample)
y = []
for i in X:
    y.append(sin(i) / abs(i))
"""
"""
x = np.concatenate([np.linspace(-10, -1, n_sample / 2), np.linspace(1, 10, n_sample / 2)])
y = []
for i in X:
    y.append(1 / (i * i * i))
"""
rbfnet = RBFNet(lr=1e-2, k=2)
rbfnet.fit(X, y)

y_pred = rbfnet.predict(X)
plt.figure(figsize=(8.5, 6.5))
plt.plot(X, y, '-o', label='true')
plt.plot(X, y_pred, '-o', label='RBF-Net')
plt.legend()

plt.tight_layout()
plt.show()
