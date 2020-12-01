import numpy as np
from scipy.linalg import sqrtm

import sys
sys.path.append("../../models")
from cpca import CPCA
from pcpca import PCPCA




if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    import matplotlib
    font = {'size'   : 15}

    matplotlib.rc('font', **font)

    # Try this out with fake covariance matrices
    cov = [
        [2.7, 2.6],
        [2.6, 2.7]
    ]

    

    def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

    # Vary gamma and plot what happens
    # We expect that gamma equal to 0 recovers PCA on X
    n = 100
    m_vals = [10, 100, 200, 1000]

    k = 1
    gamma = 0.9
    plt.figure(figsize=(len(m_vals) * 7, 12))
    plt.suptitle("Gamma={}".format(gamma))
    for ii in range(len(m_vals)):

        print(ii)

        m = m_vals[ii]

        # Generate data
        Y = multivariate_normal.rvs([0, 0], cov, size=m)
        Xa = multivariate_normal.rvs([-1, 1], cov, size=n//2)
        Xb = multivariate_normal.rvs([1, -1], cov, size=n//2)
        X = np.concatenate([Xa, Xb], axis=0)

        X, Y = X.T, Y.T



        #### CPCA
        cpca = CPCA(gamma=gamma, n_components=k)
        cpca.fit(X, Y)

        plt.subplot(2, len(m_vals), ii+1)
        plt.title("m={}".format(m))
        plt.scatter(X[0, :], X[1, :], alpha=0.5, label="X (target)")
        plt.scatter(Y[0, :], Y[1, :], alpha=0.5, label="Y (background)")
        plt.legend()
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])

        origin = np.array([[0], [0]])  # origin point
        abline(slope=cpca.W[1, 0] / cpca.W[0, 0], intercept=0)

        if ii == 0:
            plt.ylabel("CPCA        ", rotation=0)

        import ipdb; ipdb.set_trace()




        #### PCPCA
        pcpca = PCPCA(gamma=n/m*gamma, n_components=k)
        pcpca.fit(X, Y)

        plt.subplot(2, len(m_vals), len(m_vals)+ii+1)
        plt.title("m={}".format(m))
        plt.scatter(X[0, :], X[1, :], alpha=0.5, label="X (target)")
        plt.scatter(Y[0, :], Y[1, :], alpha=0.5, label="Y (background)")
        plt.legend()
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])

        origin = np.array([[0], [0]])  # origin point
        abline(slope=pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0], intercept=0)

        if ii == 0:
            plt.ylabel("PCPCA        ", rotation=0)

        # import ipdb; ipdb.set_trace()




    plt.savefig("../../plots/cpca_varying_samplesizes.png")
    plt.show()




