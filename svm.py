import numpy as np
from sklearn import svm
from math import sin
import matplotlib.pyplot as plt

X = np.linspace(-3,3,600).reshape(-1, 1)
y = []
#print(x)
for i in X.flat:
    #print(i)
    y.append(sin(i) / abs(i))

clf = svm.SVR(kernel="rbf")
clf.fit(X, y)
x = X.reshape(-1)
y_pred = clf.predict(X)
plt.plot(x, y, '-o', label='true')
plt.plot(x, y_pred, '-o', label='BP-Net')
plt.legend()

plt.tight_layout()
plt.show()