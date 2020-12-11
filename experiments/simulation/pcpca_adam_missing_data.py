import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
import functools
import tensorflow as tf

import sys
from sklearn.decomposition import PCA
from numpy.linalg import slogdet
sys.path.append("../../models")
from pcpca import PCPCA
from cpca import CPCA

inv = np.linalg.inv

n, m = 100, 100
p = 10
k = 2
ky = 2
zx = np.random.normal(0, 1, size=(k, n))
zy = np.random.normal(0, 1, size=(ky, m))
W_true = np.random.normal(0, 10, size=(p, k))
S_true = np.random.normal(0, 1, size=(p, ky))
sigma2_true = 3

X_full = W_true @ zx + np.random.normal(0, np.sqrt(sigma2_true), size=(p, n))
Y_full = S_true @ zy + np.random.normal(0, np.sqrt(sigma2_true), size=(p, m))

## Test data
zx_test = np.random.normal(0, 1, size=(k, n))
zy_test = np.random.normal(0, 1, size=(ky, m))
X_test = W_true @ zx_test + np.random.normal(0, np.sqrt(sigma2_true), size=(p, n))
Y_test = S_true @ zy_test + np.random.normal(0, np.sqrt(sigma2_true), size=(p, m))

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

    # import ipdb; ipdb.set_trace()
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


gamma = 0.2
missing_p = 0.2

X = X_full.copy()
Y = Y_full.copy()
missing_mask_X = np.random.choice([0, 1], p=[1-missing_p, missing_p], size=(p, n)).astype(bool)
missing_mask_Y = np.random.choice([0, 1], p=[1-missing_p, missing_p], size=(p, m)).astype(bool)

X[missing_mask_X] = np.nan
Y[missing_mask_Y] = np.nan

alpha = 0.1
beta_1 = 0.9
beta_2 = 0.999						#initialize the values of the parameters
epsilon = 1e-8
m_t = 0 
v_t = 0 
m_t_sigma2 = 0 
v_t_sigma2 = 0 
t = 0
W = np.random.normal(size=(X.shape[0], k))
sigma2 = 1.0
n_iter = 300

ll_trace = []
for iter_num in range(n_iter):					#till it gets converged
	t+=1
	g_t_W, g_t_sigma2 = grads(X, Y, W, sigma2, gamma)		#computes the gradient of the stochastic function

	## W
	m_t = beta_1*m_t + (1-beta_1)*g_t_W	#updates the moving averages of the gradient
	v_t = beta_2*v_t + (1-beta_2)*(g_t_W*g_t_W)	#updates the moving averages of the squared gradient
	m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
	v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates
	W_prev = W								
	W = W + (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)	#updates the parameters
	# if(W == W_prev):		#checks if it is converged or not
	# 	break

	## sigma2
	m_t_sigma2 = beta_1*m_t_sigma2 + (1-beta_1)*g_t_sigma2	#updates the moving averages of the gradient
	v_t_sigma2 = beta_2*v_t_sigma2 + (1-beta_2)*(g_t_sigma2*g_t_sigma2)	#updates the moving averages of the squared gradient
	m_cap = m_t_sigma2/(1-(beta_1**t))		#calculates the bias-corrected estimates
	v_cap = v_t_sigma2/(1-(beta_2**t))		#calculates the bias-corrected estimates
	sigma2_prev = sigma2								
	sigma2 = sigma2 + (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)	#updates the parameters
	# if(sigma2 == sigma2_prev):		#checks if it is converged or not
	# 	break

	# import ipdb; ipdb.set_trace()
	ll = log_likelihood(X, Y, W, sigma2, gamma)
	ll_trace.append(ll)

	
	if iter_num % 50 == 0:
		print("LL: {}".format(ll))
		print(sigma2)
		ll_test = log_likelihood_fg(X_test, W, sigma2, gamma)
		print(ll_test)

print(sigma2)
plt.plot(ll_trace)
plt.show()
import ipdb; ipdb.set_trace()










