import matplotlib
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from pcpca import PCPCA
import numpy as np


font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


# Generate data
n, m = 200, 200

cov = [[2.7, 2.6], [2.6, 2.7]]
X = multivariate_normal.rvs([0, 0], cov, size=m)

Ya = multivariate_normal.rvs([-1, 1], cov, size=n // 2)
Yb = multivariate_normal.rvs([1, -1], cov, size=n // 2)
Y = np.concatenate([Ya, Yb], axis=0)


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--")


cross_cov = np.zeros((2, 2))
for ii in range(n):
    for jj in range(m):
        cross_cov += np.outer(X[ii, :], Y[jj, :])

cross_cov /= (n + m)

beta = np.linalg.inv(X.T @ X) @ cross_cov


preds = X @ beta
pred_cross_cov = np.zeros((2, 2))
for ii in range(n):
    for jj in range(m):
        pred_cross_cov += np.outer(preds[ii, :], Y[jj, :])

C = Y.T @ Y - pred_cross_cov

plt.scatter(X[:, 0], X[:, 1], label="bg")
plt.scatter((X @ beta)[:, 0], (X @ beta)[:, 1], label="Pred bg")
plt.legend()
plt.show()

import ipdb; ipdb.set_trace()
