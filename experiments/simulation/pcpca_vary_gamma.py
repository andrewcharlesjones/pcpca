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
Y = multivariate_normal.rvs([0, 0], cov, size=m)

Xa = multivariate_normal.rvs([-1, 1], cov, size=n // 2)
Xb = multivariate_normal.rvs([1, -1], cov, size=n // 2)
X = np.concatenate([Xa, Xb], axis=0)

X, Y = X.T, Y.T


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--")


# Vary gamma and plot what happens
# We expect that gamma equal to 0 recovers PCA on X
gamma_range = [0, 0.2, 0.6, 0.9]
k = 1
plt.figure(figsize=(len(gamma_range) * 7, 7))
for ii, gamma in enumerate(gamma_range):
    pcpca = PCPCA(gamma=gamma, n_components=k)
    pcpca.fit(X, Y)

    plt.subplot(1, len(gamma_range), ii + 1)
    if gamma == 0:
        plt.title(r"$\gamma^\prime$={}  (PPCA)".format(gamma))
    else:
        plt.title(r"$\gamma^\prime$={}".format(gamma))
    plt.scatter(
        X[0, : n // 2],
        X[1, : n // 2],
        alpha=0.5,
        label="Foreground group 1",
        s=80,
        color="green",
    )
    plt.scatter(
        X[0, n // 2 :],
        X[1, n // 2 :],
        alpha=0.5,
        label="Foreground group 2",
        s=80,
        color="orange",
    )
    plt.scatter(Y[0, :], Y[1, :], alpha=0.5, label="Background", s=80, color="gray")
    plt.legend(prop={"size": 20})
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])

    origin = np.array([[0], [0]])  # origin point
    abline(slope=pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0], intercept=0)

    print(pcpca.W_mle)
plt.tight_layout()
plt.savefig("../../plots/simulated/pcpca_vary_gamma.png")
plt.show()
