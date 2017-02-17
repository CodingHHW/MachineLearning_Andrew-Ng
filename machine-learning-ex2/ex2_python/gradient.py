import numpy as np
from sigmoid import sigmoid


def gradient(theta, x, y):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    grad = np.zeros((n, 1))
    h = sigmoid(x.dot(theta))
    grad = ((x.T).dot(h - y)) / m
    return grad.flatten()
