# -*- coding: UTF-8 -*-
#感知机模型比较简单。感知机只有w,b两个参数，存在一个超平面使得正负样本完全分开。
# 感知机训练过程就是求解w和b的过程，w是个n维向量，而b是一个标量。为了保证收敛，我们把w,b初始化为0
#
import numpy as np
from utils import gen_two_clusters


class Perceptron:
    def __init__(self):
        self._w = self._b = None

    def fit(self, x, y, lr=0.01, epoch=1000):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self._w = np.zeros(x.shape[1])
        self._b = 0.
        for _ in range(epoch):
            err = -y * self.predict(x,False )
            idx = np.argmax(err)
            if err[idx] < 0:
                break
            delta = lr * y[idx]
            self._w += delta * x[idx]
            self._b += delta

    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        y_pred = x.dot(self._w) + self._b
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)

x, y = gen_two_clusters()
perceptron = Perceptron()
perceptron.fit(x, y)
print(u"准确率：{:8.6} %".format((perceptron.predict(x) == y).mean() * 100))

from utils import visualize2d

visualize2d(perceptron, x, y)
visualize2d(perceptron, x, y, True)
'''
梯度下降
梯度下降只考虑两个问题：
求损失函数的下降梯度（求导）
将参数沿梯度反向走一步

感知机的损失函数：
L(x,y)=max(-y(wx+b),0)
易知，只有错误分类的点，才会贡献损失。

所以我们只需挑选损失最大的点，用他来计算梯度，进行梯度下降，如果这个点都正确分类了，说明所有点都正确分类了。
'''


