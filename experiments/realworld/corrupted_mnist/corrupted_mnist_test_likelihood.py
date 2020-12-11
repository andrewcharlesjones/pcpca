from pcpca import PCPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
from scipy.stats import multivariate_normal
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import sys
sys.path.append("../../../models")


N_COMPONENTS = 2

if __name__ == "__main__":

    # Load data
    data_dir = "../../../data/corrupted_mnist"
    X = np.load(pjoin(data_dir, "foreground.npy"))
    Y = np.load(pjoin(data_dir, "background.npy"))
    X_labels = np.load(pjoin(data_dir, "foreground_labels.npy"))
    digits_test = np.load(pjoin(data_dir, "mnist_digits_test.npy"))

    X_mean, Y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
    X = (X - X_mean) / np.std(X, axis=0)
    Y = (Y - Y_mean) / np.std(Y, axis=0)

    digits_test_mean = np.mean(digits_test, axis=0)
    digits_test_std = np.std(digits_test, axis=0)
    digits_test -= digits_test_mean
    variable_idx = np.where(digits_test_std > 0)[0]
    digits_test[:, variable_idx] /= digits_test_std[variable_idx]

    X, Y = X.T, Y.T
    digits_test = digits_test.T

    n, m = X.shape[1], Y.shape[1]

    # Get test log-likelihood across a range of gamma
    gamma_range = np.linspace(0, 0.9, 20)

    test_lls = []
    for gamma in gamma_range:
        pcpca = PCPCA(gamma=gamma*n/m, n_components=N_COMPONENTS)
        pcpca.fit(X, Y)
        test_ll = pcpca._log_likelihood(
            digits_test, pcpca.W_mle, pcpca.sigma2_mle)
        test_lls.append(test_ll)

    import matplotlib
    font = {'size': 20}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['text.usetex'] = True

    plt.figure(figsize=(7, 5))
    plt.plot(gamma_range, test_lls, 'o-')
    plt.xlabel(r'$\gamma^\prime$')
    plt.ylabel("Test log-likelihood")
    ax = plt.gca()
    plt.text(x=-0.03, y=-0.2, s="(PPCA)", transform=ax.transAxes)
    plt.title("Likelihood of MNIST digits")
    plt.tight_layout()
    plt.savefig("../../../plots/corrupted_mnist/mnist_test_likelihood.png")
    plt.show()
