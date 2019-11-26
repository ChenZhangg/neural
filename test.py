import numpy as np
from numpy import linalg as LA

C = 200  # 样本数
x = np.arange(C)
s1 = 2 * np.sin(0.02 * np.pi * x)  # 正弦信号

a = np.linspace(-2, 2, 25)
s2 = np.array([a, a, a, a, a, a, a, a]).reshape(200, )  # 锯齿信号
s3 = np.array(20 * (5 * [2] + 5 * [-2]))  # 方波信号
s4 = 4 * (np.random.random([1, C]) - 0.5).reshape(200, )  # 随机信号

#print(s4)
s = np.array([s1, s2, s3, s4])  # 合成信号
#print(s[3, :])
ran = 2 * np.random.random([4, 4])  # 随机矩阵
mix = ran.dot(s)  # 混合信号

average=np.mean(mix, axis=1)
print(average)