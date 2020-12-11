import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal


class PCPCA:

    def __init__(self, n_components=2, gamma=0.5):
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

        self.sigma2_mle = sigma2_mle
        self.W_mle = W_mle

    def transform(self, X, Y, use_em=False):
        """Embed data using fitted model.
        """
        if use_em:
            t = self.W_em.T @ X, self.W_em.T @ Y
        else:
            t = self.W_mle.T @ X, self.W_mle.T @ Y
        return t

    def fit_transform(self, X, Y, use_em=False):
        if use_em:
            # self.fit_em_missing_data(X, Y)
            self.fit_em(X, Y)
            t = self.transform(X, Y, use_em=True)
        else:
            self.fit(X, Y)
            t = self.transform(X, Y, use_em=False)
        return t

    def sample(self):
        """Sample from the fitted model.
        """
        pass

    def get_gamma_bound(self, X, Y):
        """Compute the upper bound on gamma such that sigma2 > 0.
        """
        p = X.shape[0]
        Cx = self._compute_sample_covariance(X)
        Cy = self._compute_sample_covariance(Y)
        Cx_eigvals = -np.sort(-np.linalg.eigvals(Cx))
        Cy_eigvals = -np.sort(-np.linalg.eigvals(Cy))

        gamma_bound = np.sum(Cx_eigvals[self.k-1:]) / \
            ((p - self.k) * Cy_eigvals[0])
        return gamma_bound

    def _compute_sample_covariance(self, data):
        """Compute sample covariance where data is a p x n matrix.
        """
        n = data.shape[1]
        cov = 1 / n * data @ data.T
        return cov

    def gradient_descent_missing_data(self, X, Y, n_iter=500, verbose=True):
        inv = np.linalg.inv
        slogdet = np.linalg.slogdet

        def make_L(X):
            p = X.shape[0]
            unobserved_idx = np.where(np.isnan(X))[0]
            observed_idx = np.setdiff1d(np.arange(p), unobserved_idx)
            L = np.zeros((observed_idx.shape[0], p))
            for ii, idx in enumerate(observed_idx):
                L[ii, idx] = 1
            return L

        def grads(X, Y, W, sigma2, gamma):
            p, n = X.shape
            m = Y.shape[1]

            # Indication matrices
            Ls = [make_L(X[:, ii]) for ii in range(n)]
            Ms = [make_L(Y[:, ii]) for ii in range(m)]

            As = [Ls[ii] @ (W @ W.T + sigma2 * np.eye(p)) @ Ls[ii].T for ii in range(n)]
            Bs = [Ms[ii] @ (W @ W.T + sigma2 * np.eye(p)) @ Ms[ii].T for ii in range(m)]

            running_sum_W_X = np.zeros((p, p))
            running_sum_sigma2_X = 0
            for ii in range(n):
                L, A = Ls[ii], As[ii]
                x = L @ np.nan_to_num(X[:, ii], nan=0)
                Di = L.shape[0]
                A_inv = inv(A)

                curr_summand_W = L.T @ A_inv @ (np.eye(Di) - np.outer(x, x) @ A_inv) @ L
                running_sum_W_X += curr_summand_W

                curr_summand_sigma2 = np.trace(A_inv @ L @ L.T) - np.trace(A_inv @ np.outer(x, x) @ A_inv @ L @ L.T)
                running_sum_sigma2_X += curr_summand_sigma2

            running_sum_W_Y = np.zeros((p, p))
            running_sum_sigma2_Y = 0
            for jj in range(m):
                M, B = Ms[jj], Bs[jj]
                y = M @ np.nan_to_num(Y[:, jj], nan=0)
                Ej = M.shape[0]
                B_inv = inv(B)

                curr_summand_W = M.T @ B_inv @ (np.eye(Ej) - np.outer(y, y) @ B_inv) @ M
                running_sum_W_Y += curr_summand_W

                curr_summand_sigma2 = np.trace(B_inv @ M @ M.T) - np.trace(B_inv @ np.outer(y, y) @ B_inv @ M @ M.T)
                running_sum_sigma2_Y += curr_summand_sigma2

            W_grad = -(running_sum_W_X - gamma * running_sum_W_Y) @ W
            sigma2_grad = -0.5 * running_sum_sigma2_X + gamma/2.0 * running_sum_sigma2_Y

            return W_grad, sigma2_grad

        def log_likelihood(X, Y, W, sigma2, gamma):
            p, n = X.shape
            m = Y.shape[1]
            Ls = [make_L(X[:, ii]) for ii in range(n)]
            Ms = [make_L(Y[:, ii]) for ii in range(m)]

            As = [Ls[ii] @ (W @ W.T + sigma2 * np.eye(p)) @ Ls[ii].T for ii in range(n)]
            Bs = [Ms[ii] @ (W @ W.T + sigma2 * np.eye(p)) @ Ms[ii].T for ii in range(m)]

            running_sum_X = 0
            for ii in range(n):
                L = Ls[ii]
                A = As[ii]
                x = L @ np.nan_to_num(X[:, ii], nan=0)
                Di = L.shape[0]
                A_inv = inv(A)

                curr_summand = Di * np.log(2 * np.pi) + slogdet(A)[1] + np.trace(A_inv @ np.outer(x, x))
                running_sum_X += curr_summand

            running_sum_Y = 0
            for ii in range(m):
                M = Ms[ii]
                B = Bs[ii]
                y = M @ np.nan_to_num(Y[:, ii], nan=0)
                Ei = M.shape[0]
                B_inv = inv(B)

                curr_summand = Ei * np.log(2 * np.pi) + slogdet(B)[1] + np.trace(B_inv @ np.outer(y, y))
                running_sum_Y += curr_summand

            LL = -0.5 * running_sum_X + gamma/2.0 * running_sum_Y

            return LL

        X_col_means = np.nanmean(X.T, axis=0)
        Y_col_means = np.nanmean(Y.T, axis=0)

        # Find indices that you need to replace
        inds_X = np.where(np.isnan(X.T))
        inds_Y = np.where(np.isnan(Y.T))

        # Place column means in the indices. Align the arrays using take
        X_copy, Y_copy = X.copy(), Y.copy()
        X_copy, Y_copy = X_copy.T, Y_copy.T
        X_copy[inds_X] = np.take(X_col_means, inds_X[1])
        Y_copy[inds_Y] = np.take(Y_col_means, inds_Y[1])
        X_copy -= X_copy.mean(0)
        Y_copy -= Y_copy.mean(0)
        X_copy, Y_copy = X_copy.T, Y_copy.T

        pcpca_init = PCPCA(gamma=self.gamma, n_components=self.k)
        pcpca_init.fit(X_copy, Y_copy)
        W, sigma2 = pcpca_init.W_mle, 2  # pcpca_init.sigma2_mle

        # Adam
        alpha = 0.01
        beta_1 = 0.9
        beta_2 = 0.999  # initialize the values of the parameters
        epsilon = 1e-8
        m_t = 0
        v_t = 0
        m_t_sigma2 = 0
        v_t_sigma2 = 0
        t = 0
        # W = np.random.normal(size=(X.shape[0], self.k))
        # sigma2 = 3.0

        ll_trace = []
        ll_last = 0
        for iter_num in range(n_iter):  # till it gets converged
            t += 1
            # computes the gradient of the stochastic function
            g_t_W, g_t_sigma2 = grads(X, Y, W, sigma2, self.gamma)

            # W
            # updates the moving averages of the gradient
            m_t = beta_1*m_t + (1-beta_1)*g_t_W
            # updates the moving averages of the squared gradient
            v_t = beta_2*v_t + (1-beta_2)*(g_t_W*g_t_W)
            # calculates the bias-corrected estimates
            m_cap = m_t/(1-(beta_1**t))
            # calculates the bias-corrected estimates
            v_cap = v_t/(1-(beta_2**t))
            W_prev = W
            # updates the parameters
            W = W + (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)

            # sigma2
            # updates the moving averages of the gradient
            m_t_sigma2 = beta_1*m_t_sigma2 + (1-beta_1)*g_t_sigma2
            # updates the moving averages of the squared gradient
            v_t_sigma2 = beta_2*v_t_sigma2 + (1-beta_2)*(g_t_sigma2*g_t_sigma2)
            # calculates the bias-corrected estimates
            m_cap = m_t_sigma2/(1-(beta_1**t))
            # calculates the bias-corrected estimates
            v_cap = v_t_sigma2/(1-(beta_2**t))
            sigma2_prev = sigma2
            # updates the parameters
            sigma2 = sigma2 + (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)

            # Threshold
            sigma2 = max(sigma2, 1e-4)

            if verbose and (iter_num % 20) == 0:
                ll = log_likelihood(X, Y, W, sigma2, self.gamma)
                if np.abs(ll - ll_last) < 0.1:
                    break
                ll_last = ll
                print("Iter: {} \t LL: {}".format(iter_num, round(ll, 2)))

        return W, sigma2

    def _create_permuation_matrix(self, idx):

        P = np.zeros((idx.shape[0], idx.shape[0]))
        for ii, curr_idx in enumerate(idx):
            P[ii, curr_idx] = 1
        return P

    def _compute_Exxi(self, x, A):

        # Arrange data into observed and unobserved
        observed_idx = np.where(~np.isnan(x))[0]
        unobserved_idx = np.setdiff1d(np.arange(self.p), observed_idx)
        n_observed, n_unobserved = len(observed_idx), len(unobserved_idx)
        xo = x[observed_idx]
        xu = x[unobserved_idx]

        # Put A into block matrix form
        A_sorted_idx = np.concatenate([observed_idx, unobserved_idx])
        A = A[A_sorted_idx, :][:, A_sorted_idx]

        # Pull out blocks
        Aoo = A[:n_observed, :n_observed]
        Auo = A[n_observed:, :n_observed]
        Aou = Auo.T
        Auu = A[n_observed:, n_observed:]

        # Compute Mi/Mj
        Aoo_inv = np.linalg.inv(Aoo)
        Ai_lowerright = Auu - Auo @ Aoo_inv @ Aou
        Ai = np.block([
            [np.zeros((n_observed, n_observed)),
             np.zeros((n_observed, n_unobserved))],
            [np.zeros((n_unobserved, n_observed)),    Ai_lowerright]])

        # Compute mui/muj
        mui = np.concatenate([xo, Auo @ Aoo_inv @ xo])

        # Compute expectation of outer product
        Exxi = Ai + np.outer(mui, mui)

        # Permute back to original indices
        P = self._create_permuation_matrix(A_sorted_idx)
        Exxi = P @ Exxi @ P

        return Exxi

    def _compute_Co(self, X, Y, A):

        Exxis = np.zeros((self.p, self.p))
        for ii in range(self.n):
            Exxi = self._compute_Exxi(X[:, ii], A)
            Exxis += Exxi

        Eyyjs = np.zeros((self.p, self.p))
        for jj in range(self.m):
            Eyyj = self._compute_Exxi(Y[:, jj], A)
            Eyyjs += Eyyj

        Co = Exxis - self.gamma * Eyyjs
        return Co

    def _log_likelihood(self, X, W, sigma2):
        p = X.shape[0]
        return np.sum(multivariate_normal.logpdf(X.T, mean=np.zeros(p), cov=W @ W.T + sigma2 * np.eye(p)))

    def fit_em_missing_data(self, X, Y, n_iter=50):
        """Fit model with EM in the presence of missing data.
        Missing values must be encoded as NA.
        NOTE: THIS DOES NOT WORK CURRENTLY.
        """
        p, n, m = X.shape[0], X.shape[1], Y.shape[1]
        self.p = p
        self.n = n
        self.m = m

        # Initialize W and sigma2
        W = np.random.normal(size=(p, self.k))
        sigma2 = np.exp(np.random.normal())

        lhoods = []
        for _ in range(n_iter):

            # Compute M
            A = W @ W.T + sigma2 * np.eye(p)
            M = W.T @ W + sigma2 * np.eye(self.k)

            # Compute expecatation of differential covariance
            Co = self._compute_Co(X, Y, A)

            # Update W
            M_inv = np.linalg.inv(M)
            inv_term = (self.n - self.gamma * m) * sigma2 * M_inv + M_inv @ W.T @ Co @ W @ M_inv
            Wtilde = Co @ W @ M_inv @ np.linalg.inv(inv_term)

            # Update sigma2
            tr1 = np.trace(Co)
            tr2 = np.trace(Wtilde.T @ Wtilde * ((n - self.gamma * m) * sigma2 * M_inv + M_inv @ Wtilde.T @ Co @ Wtilde @ M_inv))
            tr3 = np.trace(Wtilde @ M_inv @ Wtilde.T @ Co)
            sigma2tilde = 1 / ((n - self.gamma * m) *
                               self.p) * (tr1 + tr2 - 2*tr3)

            # Reassign to W and sigma2
            W = Wtilde
            sigma2 = sigma2tilde

            try:
                lhood = self._log_likelihood(X, W, sigma2)
            except:
                import ipdb
                ipdb.set_trace()
            lhoods.append(lhood)
            print(lhood)

        import matplotlib.pyplot as plt
        plt.plot(lhoods)
        plt.show()

        self.W_em = W
        self.sigma2_em = sigma2

    def fit_em(self, X, Y, n_iter=50):
        """Fit model with EM in the presence of missing data.
        Missing values must be encoded as NA.
        """
        p, n, m = X.shape[0], X.shape[1], Y.shape[1]
        self.p = p
        self.n = n
        self.m = m

        # Initialize W and sigma2
        W = np.random.normal(size=(p, self.k))
        sigma2 = 0.5  # np.exp(np.random.normal())
        Cx, Cy = self._compute_sample_covariance(
            X), self._compute_sample_covariance(Y)
        C = self.n * Cx - self.gamma * self.m * Cy

        lhoods = []
        for _ in range(n_iter):

            # Compute M
            # A = W @ W.T + sigma2 * np.eye(p)
            M = W.T @ W + sigma2 * np.eye(self.k)

            # Update W
            M_inv = np.linalg.inv(M)
            inv_term = (self.n - self.gamma * m) * sigma2 * M_inv + M_inv @ W.T @ C @ W @ M_inv
            Wtilde = C @ W @ M_inv @ np.linalg.inv(inv_term)

            # Update sigma2
            tr1 = np.trace(C)
            tr2 = np.trace(Wtilde.T @ Wtilde @ ((n - self.gamma * m) * sigma2 * M_inv + M_inv @ Wtilde.T @ C @ Wtilde @ M_inv))
            tr3 = np.trace(Wtilde @ M_inv @ Wtilde.T @ C)
            sigma2tilde = 1 / ((n - self.gamma * m) *
                               self.p) * (tr1 + tr2 - 2*tr3)

            if sigma2tilde <= 0:
                import ipdb
                ipdb.set_trace()

            # Reassign to W and sigma2
            W = Wtilde
            sigma2 = sigma2tilde
            # sigma2 = max(sigma2, 0.01)

            try:
                lhood = self._log_likelihood(X, W, sigma2)
            except:
                import ipdb
                ipdb.set_trace()
            lhoods.append(lhood)
            print(lhood)

        import matplotlib.pyplot as plt
        plt.plot(lhoods)
        plt.show()
        self.W_em = W
        self.sigma2_em = sigma2


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

    # Try this out with fake covariance matrices
    cov = [
        [2.7, 2.6],
        [2.6, 2.7]
    ]

    # Generate data
    n, m = 200, 200

    miss_p_range = [0.01, 0.1, 0.5, 0.8]
    plt.figure(figsize=(len(miss_p_range) * 7, 5))

    for ii, miss_p in enumerate(miss_p_range):
        Y = multivariate_normal.rvs([0, 0], cov, size=m)
        Xa = multivariate_normal.rvs([-1, 1], cov, size=n//2)
        Xb = multivariate_normal.rvs([1, -1], cov, size=n//2)
        X = np.concatenate([Xa, Xb], axis=0)
        X -= X.mean(0)
        Y -= Y.mean(0)
        X, Y = X.T, Y.T

        missing_mask = np.random.choice(
            [0, 1], replace=True, size=(2, n), p=[1-miss_p, miss_p])
        X[missing_mask.astype(bool)] = np.nan

        gamma = 0.9
        pcpca = PCPCA(gamma=gamma, n_components=1)
        pcpca.fit_em_missing_data(X, Y)

        Xo = X[:, np.sum(np.isnan(X), axis=0) == 0]

        plt.subplot(1, len(miss_p_range), ii+1)
        plt.title("gamma = {}\nfraction missing = {}".format(gamma, miss_p))
        plt.scatter(Xo[0, :], Xo[1, :], alpha=0.5, label="X (target)")
        plt.scatter(Y[0, :], Y[1, :], alpha=0.5, label="Y (background)")
        plt.legend()
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])

        origin = np.array([[0], [0]])  # origin point
        abline(slope=pcpca.W_em[1, 0] / pcpca.W_em[0, 0], intercept=0)
    plt.show()
    import ipdb
    ipdb.set_trace()
