model_code = """
data {
	int<lower=0> n;          // number of foreground samples
	int<lower=0> m;          // number of background samples
	int<lower=0> p;          // number of features
	int<lower=0> k;          // latent dim
	matrix[m, p] Y;          // background data
	matrix[n, p] X;          // foreground data
	real<lower=0> gamma;     // foreground data
}
transformed data {
	matrix[p, p] C;
	C = 1.0 * X' * X - gamma * 1.0 * Y' * Y;
}
parameters {
	matrix[k, p] W;
	real<lower=0.1> sigma2;
}
transformed parameters {
	matrix[p, p] A;
	A = W' * W + sigma2 * diag_matrix(rep_vector(1, p));
}
model {
	target += -(n - gamma * m) * 0.5 * log_determinant(A);
	target += - 0.5 * trace(inverse(A) * C);
}
"""


from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from pcpca import PCPCA
import numpy as np
import pandas as pd
import pystan
from hashlib import md5
from os.path import join as pjoin
import pickle
import os
import matplotlib


font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", c="blue", alpha=1)


beta = 0.5
gamma = 0.3
k = 1
ns = (np.array([10, 30, 50, 100, 200]) / 2.0).astype(int)
num_reps = 20
frac_exceeding_list = np.empty((num_reps, len(ns)))
frac_exceeding_list[:] = np.nan
divs_list = []

for rep in range(num_reps):

    for n_idx, n in enumerate(ns):

        # Generate data
        m = n
        p = 2

        # cov = [
        #     [2.7, 2.6],
        #     [2.6, 2.7]
        # ]
        # Y = multivariate_normal.rvs([0, 0], cov, size=m)

        # Xa = multivariate_normal.rvs([-2, 2], cov, size=n//2)
        # Xb = multivariate_normal.rvs([2, -2], cov, size=n//2)
        # X = np.concatenate([Xa, Xb], axis=0)

        cov_bg = np.array([[2.7, 2.6], [2.6, 2.7]])
        Y = multivariate_normal.rvs(np.zeros(2), cov_bg, size=m)
        cov_fg = np.array([[2.7, 0.6], [0.6, 2.7]])
        X = multivariate_normal.rvs(np.zeros(2), cov_fg, size=n)

        # plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5, label="Background", s=50, color="gray")
        # plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Background", s=50, color="red")
        # plt.show()

        def compute_risk(W, sigma2):
            A = np.outer(W, W) + sigma2 * np.eye(p)
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
        sigma2_star = 1 / (beta - (1 - beta) * gamma) * eigvals[1]
        W_star = eigvecs[:, 0] * np.sqrt(
            1 / (beta - (1 - beta) * gamma) * eigvals[0] - sigma2_star
        )

        risk_star = compute_risk(W_star, sigma2_star)

        pcpca_data = {"n": n, "m": m, "p": p, "k": k, "X": X, "Y": Y, "gamma": gamma}

        ## Load model
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
        fit = sm.sampling(data=pcpca_data, iter=3000, warmup=2900, chains=4)

        rhat_vals = fit.summary()["summary"][:, -1]
        if np.sum(rhat_vals < 0.9) or np.sum(rhat_vals > 1.1):
            continue

        # Get samples
        W_list = np.squeeze(fit.extract()["W"])
        sigma2_list = fit.extract()["sigma2"]

        # Compute divergences
        div_list = np.zeros(len(sigma2_list))
        for ii in range(len(sigma2_list)):
            div = compute_divergence(W_list[ii, :], sigma2_list[ii])
            div_list[ii] = div

        divs_list.append(div_list)
        # plt.hist(div_list)
        # plt.show()

        risk_list = np.zeros(len(sigma2_list))
        for ii in range(len(sigma2_list)):
            risk = compute_risk(W_list[ii, :], sigma2_list[ii])
            risk_list[ii] = risk

        # epsilon = np.log(n + m) / ((n + m)**(0.5))
        epsilon = np.log(n + m) / (2 * (n + m) ** (0.5))
        # epsilon = ((n + m)**(-0.5))

        num_exceeding = np.sum(div_list > epsilon)
        frac_exceeding = 1.0 * num_exceeding / len(div_list)
        frac_exceeding_list[rep, n_idx] = frac_exceeding

        # plt.hist(div_list, 30)
        # plt.axvline(epsilon)
        # plt.show()

        # import ipdb; ipdb.set_trace()


# divs_df = pd.DataFrame(np.array(divs_list).T, columns=ns*2)
# div_means = np.mean(divs_df, 0)

# plt.xscale('log')
# xs = np.logspace(0, 5, 100)
# plt.plot(xs, xs**-0.5)

# divs_df.boxplot(positions=ns*2)
# plt.show()
# # import ipdb; ipdb.set_trace()


# divs_df = pd.melt(divs_df)
# divs_df['variable'] = divs_df.variable.values.astype(int)

# xs = np.logspace(0, 4, 100)
# plt.plot(xs, xs**-0.5)
# plt.scatter(ns*2, div_means, s=30)

# # sns.boxplot(data=divs_df, x="variable", y="value")

# plt.xscale('log')
# plt.xlabel(r'$n$')
# plt.ylabel(r'Mean $d(\theta, \theta^*)$')
# plt.tight_layout()
# plt.show()

results_df = pd.DataFrame(frac_exceeding_list, columns=ns * 2)
results_df = pd.melt(results_df)

plt.figure(figsize=(7, 5))
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel(r"$n$")
# plt.ylabel(r'Fraction $d(\theta, \theta^*) > n^{-1/2}$')
plt.ylabel(r"Fraction $d(\theta, \theta^*) > \frac{\log n}{2 n^{1/2}}$")
plt.tight_layout()
plt.savefig("../../../plots/simulated/gibbs_rate.png")
plt.show()

import ipdb

ipdb.set_trace()
