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
from scipy.optimize import minimize

inv = np.linalg.inv

n = 100
p = 10
k = 3
z = np.random.normal(0, 1, size=(k, n))
W_true = np.random.normal(0, 10, size=(p, k))
sigma2_true = 1

X = W_true @ z + np.random.normal(scale=np.sqrt(sigma2_true), size=(p, n))


def log_likelihood(X, W, sigma2):
	p, n = X.shape
	A = W @ W.T + sigma2 * np.eye(p)
	LL = -0.5 * p * n * np.log(2 * np.pi) - 0.5 * n * slogdet(A)[1] - 0.5 * n * np.trace(inv(A) @ X @ X.T)
	return LL

def log_likelihood_terms(X, W, sigma2):
	p, n = X.shape
	A = W @ W.T + sigma2 * np.eye(p)
	term1 = - 0.5 * n * slogdet(A)[1]
	term2 =  - 0.5 * n * np.trace(inv(A) @ X @ X.T)
	# import ipdb; ipdb.set_trace()
	return term1, term2


# def log_likelihood(X, W, sigma2):
# 	p, n = X.shape
# 	evidence_cov = W @ W.T + sigma2 * np.eye(p)
# 	ll = multivariate_normal.logpdf(X.T, mean=np.zeros(p), cov=evidence_cov, allow_singular=True)
# 	return np.sum(ll)

def to_vector(W, sigma2):
    assert W.shape == (p, k)
    return np.concatenate([W.flatten(), [sigma2]])

def to_arguments(vec):
    assert vec.shape == (p*k + 1,)
    return vec[:p*k].reshape(p, k), vec[-1]

def loss(x): 
    W, sigma2 = to_arguments(x)
    return -log_likelihood(X, W, sigma2)

# fun = lambda x: -log_likelihood(X, x[0], x[1])

sigma2_range = np.linspace(0.75, 5, 1000)
liks = []
term1s = []
term2s = []
W0 = np.random.normal(size=(p, k))
# u, d, vt = np.linalg.svd(W0)
# W0 = u[:, :k]
for sigma2 in sigma2_range:
	liks.append(log_likelihood(X, W_true, sigma2))
	t1, t2 = log_likelihood_terms(X, W_true, sigma2)
	term1s.append(t1)
	term2s.append(t2)
plt.plot(sigma2_range, liks)
plt.show()
plt.plot(sigma2_range, term1s, label="term 1")
plt.plot(sigma2_range, term2s, label="term 2")
plt.legend()
plt.xlabel("sigma2")
plt.ylabel("log-likelihood")
plt.show()

W0 = np.random.normal(size=(p, k))
sigma20 = 0.1
# result = minimize(f, to_vector(W0, sigma20))

# bnds = ((None, None), (0, None))
bnds = [(None, None) if ii < p*k else (0.5, None) for ii in range(p*k + 1)]
# bnds[-1] = (0, None)
res = minimize(loss, to_vector(W0, sigma20), method='SLSQP', bounds=bnds, options={'disp': True})
# res = minimize(fun, (W0, sigma20), method='SLSQP', bounds=bnds, options={'disp': True})


res.x = to_arguments(res.x)
print("W:\n", res.x[0])
print("\nsigma2:\n", res.x[1])

print(pearsonr(res.x[0][:, 0].squeeze(), W_true.squeeze()[:, 0])[0])
import ipdb; ipdb.set_trace()
# return result



# def W_grad(X, W, sigma2):
# 	p, n = X.shape
# 	A = W @ W.T + sigma2 * np.eye(p)
# 	A_inv = inv(A)

# 	grad = -n * A_inv @ W + A_inv @ X @ X.T @ A_inv @ W
# 	return grad

# def sigma2_grad(X, W, sigma2):
# 	p, n = X.shape
# 	A = W @ W.T + sigma2 * np.eye(p)
# 	A_inv = inv(A)

# 	grad = -n/2.0 * np.trace(A_inv) + 1/2.0 * np.trace(A_inv @ X @ X.T @ A_inv)
# 	return grad


# def log_likelihood(X, W, sigma2):
# 	p, n = X.shape
# 	evidence_cov = W @ W.T + sigma2 * np.eye(p)
# 	ll = multivariate_normal.logpdf(X.T, mean=np.zeros(p), cov=evidence_cov)
# 	return np.sum(ll)

# W = np.random.normal(size=(p, k))
# sigma2 = 0.1
# # print(pearsonr(W_true.squeeze(), W.squeeze()))

# n_iter = 100
# lr_W = 0.01
# lr_sigma2 = 1e-3
# ll_trace = []
# for iter_num in range(n_iter):
# 	W += lr_W * W_grad(X, W, sigma2)
# 	sigma2 += lr_sigma2 * sigma2_grad(X, W, sigma2)
# 	print(sigma2)
# 	ll = log_likelihood(X, W, sigma2)
# 	ll_trace.append(ll)

# plt.plot(ll_trace)
# plt.xlabel("Iteration")
# plt.ylabel("Log-likelihood")
# plt.show()

# W_corrs = np.empty((k, k))
# for ii in range(k):
# 	for jj in range(k):
# 		W_corrs[ii, jj] = pearsonr(W_true[:, ii], W[:, jj])[0]


# sns.heatmap(W_corrs, center=0)
# plt.show()



# import ipdb; ipdb.set_trace()










