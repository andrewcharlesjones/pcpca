import matplotlib
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pcpca import PCPCA
import numpy as np
import pystan
from hashlib import md5
from os.path import join as pjoin
import pickle
import os

font = {'size': 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-', c="blue", alpha=.3)

ns = [30, 60, 300]
p = 2

max_n = np.max(ns)
max_m = max_n
n_mcmc_iter = 20000
n_posterior_samples = 1000
n_posterior_samples_plot = 100

cov = [
    [4, 2.6],
    [2.6, 4]
]

Y_full = multivariate_normal.rvs([0, 0], cov, size=max_m)

Xa_full = multivariate_normal.rvs([-1.4, 1.4], cov, size=max_n//2)
Xb_full = multivariate_normal.rvs([1.4, -1.4], cov, size=max_n//2)
X_full = np.concatenate([Xa_full, Xb_full], axis=0)

plt.figure(figsize=(len(ns) * 7, 6))

for ii, n in enumerate(ns):

	# Get subset of data
	m = n
	Xa = Xa_full[:n//2, :]
	Xb = Xb_full[:n//2, :]
	X = np.concatenate([Xa, Xb], axis=0)
	Y = Y_full[:m, :]
	

	pcpca_data  = {
		'n': n,
		'm': m,
		'p': p,
		'k': 1,
		'X': X,
		'Y': Y,
		'gamma': 0.85
	}

	## Load model
	with open("pcpca_gibbs.stan", 'r') as file:
		model_code = file.read()
	code_hash = md5(model_code.encode('ascii')).hexdigest()
	cache_fn = pjoin('cached_models', 'cached-model-{}.pkl'.format(code_hash))

	if os.path.isfile(cache_fn):
		print("Loading cached model...")
		sm = pickle.load(open(cache_fn, 'rb'))
	else:
		print("Saving model to cache...")
		sm = pystan.StanModel(model_code=model_code)
		with open(cache_fn, 'wb') as f:
			pickle.dump(sm, f)

	# Fit model
	n_iter = n_mcmc_iter

	# Fit and check if chains mixed
	rhat_failed = True
	while rhat_failed:
		fit = sm.sampling(data=pcpca_data, iter=n_iter, warmup=n_iter-n_posterior_samples, chains=1)
		rhat_vals = fit.summary()['summary'][:, -1]
		rhat_failed = np.sum(rhat_vals < 0.9) or np.sum(rhat_vals > 1.1)

	# Get samples
	W = np.squeeze(fit.extract()['W'])
	n_samples = W.shape[0]
	rand_idx = np.random.choice(np.arange(n_samples), n_posterior_samples_plot)
	W_samples = W[rand_idx, :]

	slopes = W_samples[:, 1] / W_samples[:, 0]

	plt.subplot(1, len(ns), ii + 1)
	plt.xlim([-7, 7])
	plt.ylim([-7, 7])
	for jj in range(len(slopes)):
		abline(slope=slopes[jj], intercept=0)


	plt.title(r'$n=m={}$'.format(n))
	plt.scatter(X[:n//2, 0], X[:n//2, 1], alpha=0.5, label="Foreground group 1", s=80, color="green")
	plt.scatter(X[n//2:, 0], X[n//2:, 1], alpha=0.5, label="Foreground group 2", s=80, color="orange")
	plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5, label="Background", s=80, color="gray")
	patch = mpatches.Patch(color='blue', label='Posterior W samples')
	plt.legend(handles=[patch], prop={'size': 15})

plt.savefig("../../../plots/simulated/w_posterior_samples.png")
plt.show()

import ipdb; ipdb.set_trace()


