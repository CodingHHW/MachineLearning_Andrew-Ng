import numpy as np


def mapFeature(X1, X2):
    degree = 6
    m = np.size(X1)
    out = np.ones((m, 1))
    for i in range(degree):
        for j in range(i):
            other = (X1**(i - j) * X2**j).reshape((m, 1))
            out = np.hstack([out, other])

    return out
