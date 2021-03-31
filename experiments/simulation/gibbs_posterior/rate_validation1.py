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
import pandas as pd
import seaborn as sns
from scipy.linalg import sqrtm

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


# ns = (np.array([20, 40, 60, 100]) / 2.0).astype(int)
ns = (np.array([50, 200, 300, 400, 500]) / 2.0).astype(int)
# ns = (np.array([160, 200, 240, 280]) / 2.0).astype(int)
gamma = 0.1
w = 1.0
beta = 0.5


max_n = np.max(ns)
max_m = max_n
n_mcmc_iter = 20000
n_posterior_samples = 1000

num_reps = 3
frac_exceeding_list = np.empty((num_reps, len(ns)))
frac_exceeding_list[:] = np.nan
divs_list = []

p = 2
k = 1

cov_bg = np.array([[1, 0.7], [0.7, 1]])
cov_fg = np.array([[1, -0.7], [-0.7, 1]])

Y_full = multivariate_normal.rvs(np.zeros(p), cov_bg, size=max_m)# + np.random.normal(scale=.01, size=(max_m, p))
X_full = multivariate_normal.rvs(np.zeros(p), cov_fg, size=max_n)# + np.random.normal(scale=.01, size=(max_n, p))

# cov = np.array([[3.7, 2.6], [3.6, 2.7]])
# cov_bg = cov

# Y_full = multivariate_normal.rvs([0, 0], cov, size=max_m)

# mua = np.array([-1, 1])
# mub = np.array([1, -1])
# cov_fg = 0.5 * np.outer(mua, mua) + 0.5 * np.outer(mub, mub) + cov
# Xa_full = multivariate_normal.rvs(mua, cov, size=max_n // 2)
# Xb_full = multivariate_normal.rvs(mub, cov, size=max_n // 2)
# X_full = np.concatenate([Xa_full, Xb_full], axis=0)

# plt.scatter(X_full[:, 0], X_full[:, 1], label="FG")
# plt.scatter(Y_full[:, 0], Y_full[:, 1], label="BG")
# plt.legend()
# plt.show()


for rep in range(num_reps):

    for n_idx, n in enumerate(ns):

        # Get subset of data
        m = n
        # Xa = Xa_full[: n // 2, :]
        # Xb = Xb_full[: n // 2, :]
        # X = np.concatenate([Xa, Xb], axis=0)
        X = X_full[np.random.choice(np.arange(max_n), n, replace=False), :]
        Y = Y_full[np.random.choice(np.arange(max_m), m, replace=False), :]

        assert X.shape[0] == n
        assert Y.shape[0] == m

        def compute_risk(W, sigma2):
            # import ipdb; ipdb.set_trace()
            A = W @ W.T + sigma2 * np.eye(p)
            A_inv = np.linalg.inv(A)
            risk = 0.5 * (beta - gamma * (1 - beta)) * np.linalg.slogdet(A)[
                1
            ] + 0.5 * np.trace(A_inv @ C)
            return risk

        def compute_divergence(W, sigma2):
            risk = compute_risk(W, sigma2)
            div = np.sqrt(risk - risk_star)
            return div

        ## Compute risk minimizer
        C = beta * cov_fg - (1 - beta) * gamma * cov_bg
        eigvals, eigvecs = np.linalg.eigh(C)
        sorted_idx = np.argsort(-eigvals)
        eigvals, eigvecs = eigvals[sorted_idx], eigvecs[:, sorted_idx]
        sigma2_star = 1 / (beta - (1 - beta) * gamma) * np.mean(eigvals[1:])
        W_star = eigvecs[:, :k] @ sqrtm(
            1 / (beta - (1 - beta) * gamma) * np.diag(eigvals[:k]) - sigma2_star * np.eye(k)
        )

        print("BEST W: ", W_star)
        # import ipdb; ipdb.set_trace()
        

        risk_star = compute_risk(W_star, sigma2_star)

        pcpca_data = {"n": n, "m": m, "p": p, "k": k, "X": X, "Y": Y, "gamma": gamma, "w": w}

        ## Load model
        with open("pcpca_gibbs.stan", "r") as file:
            model_code = file.read()
        code_hash = md5(model_code.encode("ascii")).hexdigest()
        cache_fn = pjoin("cached_models", "cached-model-{}.pkl".format(code_hash))

        if os.path.isfile(cache_fn):
            print("Loading cached model...")
            sm = pickle.load(open(cache_fn, "rb"))
        else:
            print("Saving model to cache...")
            sm = pystan.StanModel(model_code=model_code)
            with open(cache_fn, "wb") as f:
                pickle.dump(sm, f)

        # Fit model
        n_iter = n_mcmc_iter

        # Fit and check if chains mixed
        rhat_failed = True
        while rhat_failed:
            fit = sm.sampling(
                data=pcpca_data, iter=n_iter, warmup=n_iter - n_posterior_samples, chains=1 #, init=[{'W': np.array([[1.6], [-1.6]]), 'sigma2': 1}]
            )
            rhat_vals = fit.summary()["summary"][:, -1]
            rhat_failed = np.sum(rhat_vals < 0.9) or np.sum(rhat_vals > 1.1)
            # if rhat_failed:
            #     print(rhat_vals)
            #     import ipdb; ipdb.set_trace()

        # Get samples
        W_list = fit.extract()["W"]
        sigma2_list = np.squeeze(fit.extract()["sigma2"])
        # Compute divergences
        div_list = np.zeros(len(sigma2_list))
        for ii in range(len(sigma2_list)):
            div = compute_divergence(W_list[ii, :], sigma2_list[ii])
            div_list[ii] = div

        divs_list.append(div_list)


        risk_list = np.zeros(len(sigma2_list))
        for ii in range(len(sigma2_list)):
            risk = compute_risk(W_list[ii, :], sigma2_list[ii])
            # should be worse than the risk minimizer
            assert risk >= risk_star
            risk_list[ii] = risk

        epsilon = 2 * ((n + m)**(-0.5))

        num_exceeding = np.sum(div_list > epsilon)
        frac_exceeding = 1.0 * num_exceeding / len(div_list)
        frac_exceeding_list[rep, n_idx] = frac_exceeding




# sns.boxplot(data=pd.melt(pd.DataFrame(np.array(divs_list).T, columns=ns*2)), x="variable", y="value")
# plt.show()

results_df = pd.DataFrame(frac_exceeding_list, columns=ns * 2)
results_df = pd.melt(results_df)

plt.figure(figsize=(7, 5))
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel(r"$n$")
# plt.ylabel(r'Fraction $d(\theta, \theta^*) > n^{-1/2}$')
plt.ylabel(r"Fraction $d(\theta, \theta^*) > n^{-1/2}$")
plt.tight_layout()
plt.savefig("../../../plots/simulated/gibbs_rate.png")
plt.show()

import ipdb

ipdb.set_trace()

