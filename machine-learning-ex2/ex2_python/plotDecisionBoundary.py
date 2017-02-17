import matplotlib.pyplot as plt
import numpy as np
from mapFeature import mapFeature


def plotDecisionBoundary(theta, X, y):
    plt.figure()
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(X[pos[0], 1], X[pos[0], 2], color='red', marker='o')
    plt.scatter(X[neg[0], 1], X[neg[0], 2], color='blue', marker='x')
    # plt.xlim(30, 100)
    # plt.ylim(30, 100)
    # plt.xlim(-1, 1.5)
    # plt.ylim(-1, 1.5)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])

    if len(theta) <= 3:
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y)
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(u[i], v[j]).dot(theta)
        z = z.T
        plt.contour(u, v, z)

    plt.show()
