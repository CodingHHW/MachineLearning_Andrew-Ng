import numpy as np
from sigmoid import sigmoid


def gradientReg(theta, x, y, lamb):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    grad = np.zeros((n, 1))
    h = sigmoid(x.dot(theta))
    grad[0] = ((x[:, 0].T).dot(h - y)) / m
    grad[1:n] = ((x[:, 1:n].T).dot(h - y)) / m + lamb / m * theta[1:n]
    return grad.flatten()
