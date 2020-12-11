import numpy as np
from scipy.linalg import sqrtm


class CPCA:

    def __init__(self, n_components, gamma):
        """Initialize PCPCA model.
        """
        self.k = n_components
        self.gamma = gamma

    def fit(self, X, Y):
        """Fit model via maximum likelihood estimation.
        """
        assert X.shape[0] == Y.shape[0]  # Should have same number of features
        p, n, m = X.shape[0], X.shape[1], Y.shape[1]

        # Get sample covariance
        Cx, Cy = self._compute_sample_covariance(
            X), self._compute_sample_covariance(Y)

        # Differential covariance
        Cdiff = Cx - self.gamma * Cy

        # Eigendecomposition
        eigvals, U = np.linalg.eig(Cdiff)

        # Sort by eigenvalues and truncate to number of components
        sorted_idx = np.argsort(-eigvals)
        eigvals = eigvals[sorted_idx]
        U = U[:, sorted_idx]
        Lambda = np.diag(eigvals)
        Lambda, U = Lambda[:self.k, :self.k], U[:, :self.k]

        W = U @ sqrtm(Lambda)

        self.W = W

    def transform(self, X, Y):
        """Embed data using fitted model.
        """
        return self.W.T @ X, self.W.T @ Y

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X, Y)

    def sample(self):
        """Sample from the fitted model.
        """
        pass

    def get_gamma_bound(self, X, Y):
        """Compute the upper bound on gamma such that the d'th eigenvalue of C is positive.
        """
        Cx = self._compute_sample_covariance(X)
        Cy = self._compute_sample_covariance(Y)
        Cx_eigvals = -np.sort(-np.linalg.eigvals(Cx))
        Cy_eigvals = -np.sort(-np.linalg.eigvals(Cy))

        gamma_bound = Cx_eigvals[self.k - 1] / Cy_eigvals[0]
        return gamma_bound

    def _compute_sample_covariance(self, data):
        """Compute sample covariance where data is a p x n matrix.
        """
        n = data.shape[1]
        cov = 1 / n * data @ data.T
        return cov


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    import matplotlib
    font = {'size': 15}

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

        for ii, gamma in enumerate(gamma_range):
            gamma_orig = gamma
            gamma *= m/n
            cpca = CPCA(gamma=gamma, n_components=k)
            cpca.fit(X, Y)

            plt.subplot(len(n_vals), len(gamma_range), len(
                gamma_range) * sample_size_ii + ii+1)
            plt.title("Gamma = m/n*{}".format(gamma_orig))
            plt.scatter(X[0, :], X[1, :], alpha=0.5, label="X (target)")
            plt.scatter(Y[0, :], Y[1, :], alpha=0.5, label="Y (background)")
            plt.legend()
            plt.xlim([-7, 7])
            plt.ylim([-7, 7])
            if ii == 0:
                plt.ylabel("n = {}     \nm={}     ".format(n, m), rotation=0)

            origin = np.array([[0], [0]])  # origin point
            abline(slope=cpca.W[1, 0] / cpca.W[0, 0], intercept=0)
    plt.savefig("../plots/cpca_varying_samplesizes.png")
    plt.show()
