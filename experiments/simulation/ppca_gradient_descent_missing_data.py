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

inv = np.linalg.inv

n = 200
p = 2
k = 1
z = np.random.normal(0, 1, size=(k, n))
W_true = np.random.normal(0, 8, size=(p, k))
sigma2_true = 1

X_full = W_true @ z + np.random.normal(scale=np.sqrt(sigma2_true), size=(p, n))


def make_L(X):
	p = X.shape[0]
	unobserved_idx = np.where(np.isnan(X))[0]
	observed_idx = np.setdiff1d(np.arange(p), unobserved_idx)
	L = np.zeros((observed_idx.shape[0], p))
	for ii, idx in enumerate(observed_idx):
		L[ii, idx] = 1
	return L

def grads(X, W, sigma2):
	p, n = X.shape
	Ls = [make_L(X[:, ii]) for ii in range(n)]
	As = [Ls[ii] @ (W @ W.T + sigma2 * np.eye(p)) @ Ls[ii].T for ii in range(n)]

	#### W gradient
	running_sum_W = np.zeros((p, p))
	running_sum_sigma2 = 0
	for ii in range(n):
		L = Ls[ii]
		A = As[ii]
		x = L @ np.nan_to_num(X[:, ii], nan=0)
		Di = L.shape[0]
		A_inv = inv(A)

		# W
		curr_summand_W = L.T @ A_inv @ (np.eye(Di) - np.outer(x, x) @ A_inv) @ L
		running_sum_W += curr_summand_W

		# sigma2
		curr_summand_sigma2 = np.trace(A_inv @ L @ L.T) - np.trace(A_inv @ np.outer(x, x) @ A_inv @ L @ L.T)
		running_sum_sigma2 += curr_summand_sigma2
		# import ipdb; ipdb.set_trace()

	W_grad = -running_sum_W @ W
	sigma2_grad = -0.5 * running_sum_sigma2

	return W_grad, sigma2_grad

def log_likelihood(X, W, sigma2):
	p, n = X.shape
	Ls = [make_L(X[:, ii]) for ii in range(n)]
	As = [Ls[ii] @ (W @ W.T + sigma2 * np.eye(p)) @ Ls[ii].T for ii in range(n)]

	#### W gradient
	running_sum = 0
	for ii in range(n):
		L = Ls[ii]
		A = As[ii]
		x = L @ np.nan_to_num(X[:, ii], nan=0)
		Di = L.shape[0]
		A_inv = inv(A)

		curr_summand = Di * np.log(2 * np.pi) + slogdet(A)[1] + np.trace(A_inv @ np.outer(x, x))
		running_sum += curr_summand


	LL = -0.5 * running_sum

	return LL



missing_p_range = [0, 0.5, 0.99] #, 0.1, 0.5, 0.99]
n_plots = len(missing_p_range) + 1
plt.figure(figsize=(n_plots*7, 6))
plt.subplot(1, n_plots, 1)
plt.scatter(X_full[0, :], X_full[1, :])
plt.title("Data")

for ii, missing_p in enumerate(missing_p_range):

	# Mask out missing data
	X = X_full.copy()
	missing_mask = np.random.choice([0, 1], p=[1-missing_p, missing_p], size=(p, n)).astype(bool)
	X[missing_mask] = np.nan

	# Initialize parameters
	W = np.random.normal(size=(p, k))
	sigma2 = 2.0

	# Start GD
	n_iter = 400
	lr_W = 0.01
	lr_sigma2 = 1e-3
	ll_trace = []
	for iter_num in range(n_iter):
		W_grad, sigma2_grad = grads(X, W, sigma2)
		W += lr_W * W_grad
		sigma2 += lr_sigma2 * sigma2_grad
		# print(sigma2)
		if iter_num % 20 == 0:
			ll = log_likelihood(X, W, sigma2)
			print("Iter: {}, log-likelihood: {}".format(iter_num, ll))
			ll_trace.append(ll)

	# plt.plot(ll_trace)
	# plt.show()

	## Generate new data from model
	zs = np.random.normal(0, 1, size=(k, n))
	X_generated = W @ zs + np.random.normal(0, np.sqrt(sigma2), size=(p, n))

	plt.subplot(1, n_plots, ii+2)
	plt.scatter(X_generated[0, :], X_generated[1, :])
	plt.title("Generated data\n{} missing".format(missing_p))
plt.show()


import ipdb; ipdb.set_trace()










