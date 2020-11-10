import numpy as np
from scipy.linalg import sqrtm
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


    n_vals = [200, 200, 20]
    m_vals = [200, 20, 200]

    k = 1
    gamma_range = [0, 0.2, 0.5, 0.9, 0.99]
    plt.figure(figsize=(len(gamma_range) * 7, 15))
    for sample_size_ii in range(len(n_vals)):

        n, m = n_vals[sample_size_ii], m_vals[sample_size_ii]

        # Generate data
        Y = multivariate_normal.rvs([0, 0], cov, size=m)
        Xa = multivariate_normal.rvs([-1, 1], cov, size=n//2)
        Xb = multivariate_normal.rvs([1, -1], cov, size=n//2)
        X = np.concatenate([Xa, Xb], axis=0)

        X, Y = X.T, Y.T

        def abline(slope, intercept):
            """Plot a line from slope and intercept"""
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, '--')

        # Vary gamma and plot what happens
        # We expect that gamma equal to 0 recovers PCA on X
        
        
        
        for ii, gamma in enumerate(gamma_range):
            # gamma *= m/n
            pcpca = PCPCA(gamma=gamma, n_components=k)
            # import ipdb; ipdb.set_trace()
            pcpca.fit(X, Y)

            plt.subplot(len(n_vals), len(gamma_range), len(gamma_range) * sample_size_ii + ii+1)
            plt.title("Gamma = {}".format(gamma))
            plt.scatter(X[0, :], X[1, :], alpha=0.5, label="X (target)")
            plt.scatter(Y[0, :], Y[1, :], alpha=0.5, label="Y (background)")
            plt.legend()
            plt.xlim([-7, 7])
            plt.ylim([-7, 7])
            if ii == 0:
                plt.ylabel("n = {}\nm={}".format(n, m), rotation=0)

            origin = np.array([[0], [0]])  # origin point
            abline(slope=pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0], intercept=0)

    plt.suptitle("Sample size-adjusted gamma")
    plt.savefig("../plots/pcpca_varying_samplesizes.png")
    plt.show()
