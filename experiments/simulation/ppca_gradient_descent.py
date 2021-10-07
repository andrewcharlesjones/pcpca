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

inv = np.linalg.inv

n = 200
p = 10
k = 2
z = np.random.normal(0, 1, size=(k, n))
W_true = np.random.normal(0, 10, size=(p, k))
sigma2_true = 1

X = W_true @ z + np.random.normal(scale=np.sqrt(sigma2_true), size=(p, n))


def W_grad(X, W, sigma2):
    p, n = X.shape
    A = W @ W.T + sigma2 * np.eye(p)
    A_inv = inv(A)

    grad = -n * A_inv @ W + A_inv @ X @ X.T @ A_inv @ W
    return grad


def sigma2_grad(X, W, sigma2):
    p, n = X.shape
    A = W @ W.T + sigma2 * np.eye(p)
    A_inv = inv(A)

    grad = -n / 2.0 * np.trace(A_inv) + 1 / 2.0 * np.trace(A_inv @ X @ X.T @ A_inv)
    return grad


def log_likelihood(X, W, sigma2):
    p, n = X.shape
    evidence_cov = W @ W.T + sigma2 * np.eye(p)
    ll = multivariate_normal.logpdf(X.T, mean=np.zeros(p), cov=evidence_cov)
    return np.sum(ll)


W = np.random.normal(size=(p, k))
sigma2 = 2.0
# print(pearsonr(W_true.squeeze(), W.squeeze()))

n_iter = 1000
lr_W = 0.01
lr_sigma2 = 1e-3
ll_trace = []
for iter_num in range(n_iter):
    W += lr_W * W_grad(X, W, sigma2)
    sigma2 += lr_sigma2 * sigma2_grad(X, W, sigma2)
    print(sigma2)
    ll = log_likelihood(X, W, sigma2)
    ll_trace.append(ll)

plt.plot(ll_trace)
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.show()

W_corrs = np.empty((k, k))
for ii in range(k):
    for jj in range(k):
        W_corrs[ii, jj] = pearsonr(W_true[:, ii], W[:, jj])[0]


sns.heatmap(W_corrs, center=0)
plt.show()


import ipdb

ipdb.set_trace()
