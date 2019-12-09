import numpy as np
from sklearn import svm
from math import sin
from math import pi
import matplotlib.pyplot as plt

n_sample = 1000
"""
X = np.linspace(-6 * pi, 6 * pi,n_sample).reshape(-1, 1)
y = []
#print(x)
for i in X.flat:
    y.append(sin(i) / abs(i))
"""
"""
X = np.linspace(-6 * pi, 6 * pi,n_sample).reshape(-1, 1)
y = []
#print(x)
for i in X.flat:
    y.append(sin(i))
"""
X = np.concatenate([np.linspace(-10, -1, n_sample / 2), np.linspace(1, 10, n_sample / 2)]).reshape(-1, 1)
y = []
#print(x)
for i in X.flat:
    y.append(1 / (i * i * i))

clf = svm.SVR(kernel="rbf")
clf.fit(X, y)
x = X.reshape(-1)
y_pred = clf.predict(X)
plt.figure(figsize=(8.5, 6.5))
plt.plot(x, y, '-o', label='True')
plt.plot(x, y_pred, '-o', label='SVM')
plt.legend()

plt.tight_layout()
plt.show()