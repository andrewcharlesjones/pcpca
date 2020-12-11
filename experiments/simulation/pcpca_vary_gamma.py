import numpy as np
from scipy.linalg import sqrtm

import sys
sys.path.append("../../models")
from pcpca import PCPCA
from cpca import CPCA

from sklearn.decomposition import PCA




import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


import matplotlib
font = {'size'   : 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

# Try this out with fake covariance matrices


# Generate data
n, m = 200, 200

cov = [
    [2.7, 2.6],
    [2.6, 2.7]
]
Y = multivariate_normal.rvs([0, 0], cov, size=m)

Xa = multivariate_normal.rvs([-1, 1], cov, size=n//2)
Xb = multivariate_normal.rvs([1, -1], cov, size=n//2)
X = np.concatenate([Xa, Xb], axis=0)


# X = PCA(n_components=2).fit_transform(X)
# Y = PCA(n_components=2).fit_transform(Y)

X, Y = X.T, Y.T

# tmp = X
# X = Y
# Y = tmp

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

# Vary gamma and plot what happens
# We expect that gamma equal to 0 recovers PCA on X
# gamma_range = [0, 0.2, 0.5, 0.9, 0.99]
gamma_range = [0, 0.2, 0.6, 0.9]
k = 1
plt.figure(figsize=(len(gamma_range) * 7, 7))
for ii, gamma in enumerate(gamma_range):
    pcpca = PCPCA(gamma=gamma, n_components=k)
    pcpca.fit(X, Y)
    # cpca = CPCA(gamma=gamma, n_components=k)
    # cpca.fit(X, Y)

    plt.subplot(1, len(gamma_range), ii+1)
    if gamma == 0:
        # plt.title("Gamma = {} (PPCA)".format(gamma))
        plt.title(r'$\gamma^\prime$={}  (PPCA)'.format(gamma))
    else:
        # plt.title("Gamma = {}".format(gamma))
        plt.title(r'$\gamma^\prime$={}'.format(gamma))
    plt.scatter(X[0, :], X[1, :], alpha=0.5, label="X (foreground)", s=80)
    plt.scatter(Y[0, :], Y[1, :], alpha=0.5, label="Y (background)", s=80)
    # plt.legend()
    plt.legend(prop={'size': 20})
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])

    origin = np.array([[0], [0]])  # origin point
    abline(slope=pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0], intercept=0)
    # abline(slope=cpca.W[1, 0] / cpca.W[0, 0], intercept=0)

    print(pcpca.sigma2_mle)
plt.tight_layout()
plt.savefig("../../plots/simulated/pcpca_vary_gamma.png")
plt.show()
