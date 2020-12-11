import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
import sys
from sklearn.decomposition import PCA
from numpy.linalg import slogdet
sys.path.append("../../models")
from pcpca import PCPCA
from cpca import CPCA

import matplotlib
font = {'size'   : 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

inv = np.linalg.inv

n, m = 100, 100
p = 10
k = 2
ky = 2
zx = np.random.normal(0, 1, size=(k, n))
zy = np.random.normal(0, 1, size=(ky, m))
W_true = np.random.normal(0, 1, size=(p, k))
S_true = np.random.normal(0, 1, size=(p, ky))
sigma2_true = 1

X_full = W_true @ zx + np.random.normal(0, np.sqrt(sigma2_true), size=(p, n))
Y_full = S_true @ zy + np.random.normal(0, np.sqrt(sigma2_true), size=(p, m))

## Test data
zx_test = np.random.normal(0, 1, size=(k, n))
zy_test = np.random.normal(0, 1, size=(ky, m))
X_test = W_true @ zx_test + np.random.normal(0, np.sqrt(sigma2_true), size=(p, n))
Y_test = S_true @ zy_test + np.random.normal(0, np.sqrt(sigma2_true), size=(p, m))

Cx_eigvals = -np.sort(-np.linalg.eigvals(np.cov(X_full)))
Cy_eigvals = -np.sort(-np.linalg.eigvals(np.cov(Y_full)))

gamma_bound = np.sum(Cx_eigvals[k:]) / ((p - k) * Cy_eigvals[0])
print("Upper bound on gamma: {}".format(round(gamma_bound, 3)))

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def make_L(X):
    p = X.shape[0]
    unobserved_idx = np.where(np.isnan(X))[0]
    observed_idx = np.setdiff1d(np.arange(p), unobserved_idx)
    L = np.zeros((observed_idx.shape[0], p))
    for ii, idx in enumerate(observed_idx):
        L[ii, idx] = 1
    return L

def log_likelihood_fg(X, W, sigma2, gamma):
	p, n = X.shape
	Ls = [make_L(X[:, ii]) for ii in range(n)]

	As = [Ls[ii] @ (W @ W.T + sigma2 * np.eye(p)) @ Ls[ii].T for ii in range(n)]

	running_sum_X = 0
	for ii in range(n):
		L = Ls[ii]
		A = As[ii]
		x = L @ np.nan_to_num(X[:, ii], nan=0)
		Di = L.shape[0]
		A_inv = inv(A)

		curr_summand = Di * np.log(2 * np.pi) + slogdet(A)[1] + np.trace(A_inv @ np.outer(x, x))
		running_sum_X += curr_summand


	LL = -0.5 * running_sum_X

	return LL

# sigma2_range = np.linspace(1e-3, 10, 100)
# ll_list = []
# W = np.random.normal(size=(p, k))
# pcpca = PCPCA()
# for curr_sigma2 in sigma2_range:
#     curr_ll = pcpca._log_likelihood(X_full, W, curr_sigma2)
#     ll_list.append(curr_ll)
# import matplotlib.pyplot as plt
# plt.plot(sigma2_range, ll_list)
# plt.show()
# import ipdb; ipdb.set_trace()


gamma = 0.05
missing_p_range = np.arange(0, 0.8, 0.1)
n_repeats = 5
W_errors = np.empty((n_repeats, len(missing_p_range)))

for repeat_ii in range(n_repeats):
	for ii, missing_p in enumerate(missing_p_range):

		# Mask out missing data
		X = X_full.copy()
		Y = Y_full.copy()
		missing_mask_X = np.random.choice([0, 1], p=[1-missing_p, missing_p], size=(p, n)).astype(bool)
		missing_mask_Y = np.random.choice([0, 1], p=[1-missing_p, missing_p], size=(p, m)).astype(bool)

		X[missing_mask_X] = np.nan
		Y[missing_mask_Y] = np.nan

		n_obs, m_obs = np.sum(~np.isnan(X.flatten())), np.sum(~np.isnan(Y.flatten()))

		pcpca = PCPCA(gamma=gamma, n_components=k)
		W, sigma2 = pcpca.gradient_descent_missing_data(X, Y, n_iter=300)

		ll_test = log_likelihood_fg(X_test, W, sigma2, gamma)
		W_errors[repeat_ii, ii] = ll_test

		print("-" * 80)
		print("Test LL : {}".format(round(ll_test, 3)))
		print("-" * 80)
		
		# plt.scatter(W.squeeze(), W_true.squeeze())
		# plt.show()

		W = (W - np.mean(W, axis=0)) / np.std(W, axis=0)
		W_true = (W_true - np.mean(W_true, axis=0)) / np.std(W_true, axis=0)
		W_cov_true = np.cov(W_true) # W_true @ W_true.T #
		W_cov_estimated = np.cov(W) # W @ W.T #
		W_error = np.mean((W_cov_true - W_cov_estimated)**2) # pearsonr(W_cov_true.flatten(), W_cov_estimated.flatten())[0]
		# W_error = np.mean((W_cov_true / np.max(W_cov_true) - W_cov_estimated / np.max(W_cov_estimated))**2)
		print("-" * 80)
		print("W error : {}".format(round(W_error, 3)))
		print("-" * 80)
		# W_errors[repeat_ii, ii] = W_error

		# print(sigma2)
		print("\n\n")

		# Normalize
		# import ipdb; ipdb.set_trace()
		# W /= W.sum(0)
		# W_true /= W_true.sum(0)


		# W_cov_true = np.cov(W_true) # W_true @ W_true.T #
		# W_cov_estimated = np.cov(W) # W @ W.T #
		# W_error = np.mean((W_cov_true - W_cov_estimated)**2)
		# print("-" * 80)
		# print("W error : {}".format(round(W_error, 3)))
		# print("-" * 80)
		# W_errors[repeat_ii, ii] = W_error

# plt.errorbar(missing_p_range, np.mean(W_errors, axis=0), yerr=np.std(W_errors, axis=0), fmt='-o', label="PCPCA")
# plt.show()

gamma = 0.0
W_errors_ppca = np.empty((n_repeats, len(missing_p_range)))

for repeat_ii in range(n_repeats):
	for ii, missing_p in enumerate(missing_p_range):

		# Mask out missing data
		X = X_full.copy()
		Y = Y_full.copy()
		missing_mask_X = np.random.choice([0, 1], p=[1-missing_p, missing_p], size=(p, n)).astype(bool)
		missing_mask_Y = np.random.choice([0, 1], p=[1-missing_p, missing_p], size=(p, m)).astype(bool)

		X[missing_mask_X] = np.nan
		Y[missing_mask_Y] = np.nan

		n_obs, m_obs = np.sum(~np.isnan(X.flatten())), np.sum(~np.isnan(Y.flatten()))

		pcpca = PCPCA(gamma=gamma, n_components=k)
		W, sigma2 = pcpca.gradient_descent_missing_data(np.concatenate([X, Y], axis=1), Y, n_iter=300)

		ll_test = log_likelihood_fg(X_test, W, sigma2, gamma)
		W_errors_ppca[repeat_ii, ii] = ll_test

		print("-" * 80)
		print("Test LL : {}".format(round(ll_test, 3)))
		print("-" * 80)
		print("\n\n")

plt.figure(figsize=(7, 5))
plt.errorbar(missing_p_range, np.mean(W_errors, axis=0), yerr=np.std(W_errors, axis=0), fmt='-o', label="PCPCA")
plt.errorbar(missing_p_range, np.mean(W_errors_ppca, axis=0), yerr=np.std(W_errors, axis=0), fmt='-o', label="PPCA")
plt.legend()
# plt.xlabel("Fraction missing")
# plt.xlabel(r'\text{Fraction missing}')
plt.xlabel(r'Fraction missing')
# plt.ylabel("MSE")
plt.ylabel("Foreground log-likelihod (test)")
plt.title("PCPCA, missing data")
plt.tight_layout()
plt.savefig("../../plots/simulated/pcpca_missing_data.png")
plt.show()

import ipdb; ipdb.set_trace()










