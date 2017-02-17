# coding: utf-8
import numpy as np
from numpy.matlib import repmat


def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, np.size(X, axis=1)))
    sigma = np.zeros((1, np.size(X, axis=1)))

    mu = np.mean(X, axis=0)
    mu = repmat(mu, np.size(X, axis=0), 1)
    sigma = np.std(X, axis=0)
    sigma = repmat(sigma, np.size(X, axis=0), 1)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
