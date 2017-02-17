from sigmoid import sigmoid
import numpy as np


def predict(theta, X):
    m, n = X.shape
    p = np.zeros((m, 1))
    may = sigmoid(X.dot(theta))
    for i in range(m):
        if may[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p
