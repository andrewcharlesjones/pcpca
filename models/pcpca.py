import numpy as np
from scipy.linalg import sqrtm


class PCPCA:

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
        Cdiff = n * Cx - self.gamma * m * Cy


        # Eigendecomposition
        eigvals, U = np.linalg.eig(Cdiff)
        eigvals, U = np.real(eigvals), np.real(U)
        # U, D, VT = np.linalg.svd(Cdiff)
        # eigvals = D**2
        # import ipdb; ipdb.set_trace()

        # Sort by eigenvalues and truncate to number of components
        sorted_idx = np.argsort(-eigvals)
        eigvals = eigvals[sorted_idx]
        U = U[:, sorted_idx]
        Lambda = np.diag(eigvals)
        Lambda, U = Lambda[:self.k, :self.k], U[:, :self.k]

        # MLE for sigma2
        sigma2_mle = 1 / (p - self.k) * \
            np.sum(eigvals[self.k:] / (n - self.gamma * m))

        # MLE for W
        Lambda_scaled = Lambda / (n - self.gamma * m)

        W_mle = U @ sqrtm(Lambda_scaled - sigma2_mle * np.eye(self.k))
        # import ipdb; ipdb.set_trace()


        self.sigma2_mle = sigma2_mle
        self.W_mle = W_mle

    def transform(self, X, Y):
        """Embed data using fitted model.
        """
        return self.W_mle.T @ X, self.W_mle.T @ Y

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X, Y)

    def sample(self):
        """Sample from the fitted model.
        """
        pass

    def _compute_sample_covariance(self, data):
        """Compute sample covariance where data is a p x n matrix.
        """
        n = data.shape[1]
        cov = 1 / n * data @ data.T
        return cov


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    # Try this out with fake covariance matrices
    cov = [
        [2.7, 2.6],
        [2.6, 2.7]
    ]

    # Generate data
    n, m = 200, 200
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
    gamma_range = [0, 0.2, 0.5, 0.9, 0.99]
    k = 1
    plt.figure(figsize=(len(gamma_range) * 7, 5))
    for ii, gamma in enumerate(gamma_range):
        pcpca = PCPCA(gamma=gamma, n_components=k)
        # import ipdb; ipdb.set_trace()
        pcpca.fit(X, Y)

        plt.subplot(1, len(gamma_range), ii+1)
        plt.title("Gamma = {}".format(gamma))
        plt.scatter(X[0, :], X[1, :], alpha=0.5, label="X (target)")
        plt.scatter(Y[0, :], Y[1, :], alpha=0.5, label="Y (background)")
        plt.legend()
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])

        origin = np.array([[0], [0]])  # origin point
        abline(slope=pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0], intercept=0)
    plt.savefig("../plots/pcpca_vary_gamma.png")
    plt.show()
