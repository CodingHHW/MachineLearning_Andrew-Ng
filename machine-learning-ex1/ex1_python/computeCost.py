# coding: utf-8
import numpy as np


def computeCost(X, y, theta):
    ''' 代价函数 '''
    m = len(y)
    J = 0

    h = np.dot(X, theta)  # array 类型矩阵的乘法
    J = np.sum((h - y)**2) / (2 * m)  # 平方

    return J
