# coding: utf-8
import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    ''' 梯度下降算法 '''
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = np.dot(X, theta)
        theta = theta - alpha / m * np.dot((h - y).T, X).T  # 矩阵转置
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
