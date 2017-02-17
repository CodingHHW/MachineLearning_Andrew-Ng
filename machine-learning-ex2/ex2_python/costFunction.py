import numpy as np
from sigmoid import sigmoid


def costFunction(theta, x, y):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    term1 = np.log(sigmoid(x.dot(theta)))
    term2 = np.log(1 - sigmoid(x.dot(theta)))
    term1 = term1.reshape((m, 1))
    term2 = term2.reshape((m, 1))
    term = y * term1 + (1 - y) * term2
    J = -((np.sum(term)) / m)
    return J
