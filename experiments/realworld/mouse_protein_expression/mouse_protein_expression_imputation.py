import matplotlib
from pcpca import PCPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.decomposition import PCA
from numpy.linalg import slogdet
from scipy import stats

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

inv = np.linalg.inv

DATA_PATH = "../../../data/mouse_protein_expression/clean/Data_Cortex_Nuclear.csv"
N_COMPONENTS = 10

N_GD_ITER = 10
LEARNING_RATE = 1e-2
n_repeats = 3
missing_p_range = np.arange(0.1, 0.8, 0.1)


def mean_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    m, se = np.mean(data, axis=0), stats.sem(data, axis=0)
    width = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return width


# Read in data
data = pd.read_csv(DATA_PATH)


# Separate into background and foreground data
# In this case,
# background data is data from mice who did not receive shock therapty
# foreground data is from mice who did receive shock therapy


# Get names of proteins
protein_names = data.columns.values[1:78]

# Fill NAs
data = data.fillna(0)

# Background
Y_df = data[
    (data.Behavior == "C/S")
    & (data.Genotype == "Control")
    & (data.Treatment == "Saline")
]
Y = Y_df[protein_names].values
Y -= np.nanmean(Y, axis=0)
Y /= np.nanstd(Y, axis=0)
Y_full = Y.T


# Foreground
X_df = data[(data.Behavior == "S/C") & (data.Treatment == "Saline")]
X = X_df[protein_names].values
X -= np.nanmean(X, axis=0)
X /= np.nanstd(X, axis=0)
X_full = X.T


p, n = X_full.shape
_, m = Y_full.shape

# import ipdb; ipdb.set_trace()

# n_subsample = 80
# X_full = X_full[:, np.random.choice(np.arange(n), size=n_subsample, replace=False)]
# m_subsample = 80
# Y_full = Y_full[:, np.random.choice(np.arange(m), size=m_subsample, replace=False)]

# rand_idx = np.random.choice(np.arange(p), size=10)
# X_full = X_full[rand_idx, :]
# Y_full = Y_full[rand_idx, :]

p, n = X_full.shape
_, m = Y_full.shape


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--")


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

        curr_summand = (
            Di * np.log(2 * np.pi) + slogdet(A)[1] + np.trace(A_inv @ np.outer(x, x))
        )
        running_sum_X += curr_summand

    LL = -0.5 * running_sum_X

    return LL


gamma = 0.9
# missing_p_range = np.arange(0.1, 0.3, 0.1)
imputation_errors_pcpca = np.empty((n_repeats, len(missing_p_range)))
imputation_errors_ppca = np.empty((n_repeats, len(missing_p_range)))
imputation_errors_sample_means = np.empty((n_repeats, len(missing_p_range)))
imputation_errors_feature_means = np.empty((n_repeats, len(missing_p_range)))


# plt.figure(figsize=(7*len(missing_p_range), 6))
for repeat_ii in range(n_repeats):
    for ii, missing_p in enumerate(missing_p_range):

        # Mask out missing data
        X = X_full.copy()
        Y = Y_full.copy()
        missing_mask_X = np.random.choice(
            [0, 1], p=[1 - missing_p, missing_p], size=(p, n)
        ).astype(bool)
        missing_mask_Y = np.random.choice(
            [0, 1], p=[1 - missing_p, missing_p], size=(p, m)
        ).astype(bool)

        X[missing_mask_X] = np.nan
        Y[missing_mask_Y] = np.nan

        X_mean = np.nanmean(X, axis=1)
        Y_mean = np.nanmean(Y, axis=1)
        # X = (X.T - X_mean).T
        # X = (X.T / np.nanstd(X, axis=1)).T
        # Y = (Y.T - Y_mean).T
        # Y = (Y.T / np.nanstd(Y, axis=1)).T

        ### ----- Row and column means ------
        sample_means = np.nanmean(X, axis=0)
        X_imputed_sample_means = X.copy()
        X_imputed_sample_means = pd.DataFrame(X_imputed_sample_means).fillna(pd.Series(sample_means)).values
        imputation_mse = np.mean(
            (X_full[missing_mask_X] - X_imputed_sample_means[missing_mask_X]) ** 2
        )
        imputation_errors_sample_means[repeat_ii, ii] = imputation_mse

        feature_means = np.nanmean(X, axis=1)
        X_imputed_feature_means = X.copy()
        X_imputed_feature_means = pd.DataFrame(X_imputed_feature_means.T).fillna(pd.Series(feature_means)).values.T
        imputation_mse = np.mean(
            (X_full[missing_mask_X] - X_imputed_feature_means[missing_mask_X]) ** 2
        )
        print("Feature means {} missing, error: {}".format(missing_p, imputation_mse))
        imputation_errors_feature_means[repeat_ii, ii] = imputation_mse

        ### ----- PCPCA ------

        pcpca = PCPCA(gamma=gamma, n_components=N_COMPONENTS)
        W, sigma2 = pcpca.gradient_descent_missing_data(X, Y, n_iter=N_GD_ITER) #, learning_rate=LEARNING_RATE)

        X_imputed = pcpca.impute_missing_data(X)
        # X_imputed = (X_imputed.T + X_mean).T
        imputation_mse = np.mean(
            (X_full[missing_mask_X] - X_imputed[missing_mask_X]) ** 2
        )
        print("PCPCA {} missing, error: {}".format(missing_p, imputation_mse))
        imputation_errors_pcpca[repeat_ii, ii] = imputation_mse

        ### ----- PPCA ------

        X = X_full.copy()
        Y = Y_full.copy()

        X[missing_mask_X] = np.nan
        Y[missing_mask_Y] = np.nan

        fg = np.concatenate([X, Y], axis=1)
        fg_mean = np.nanmean(fg, axis=1)
        # fg = (fg.T - fg_mean).T
        # fg = (fg.T / np.nanstd(fg, axis=1)).T

        pcpca = PCPCA(gamma=0, n_components=N_COMPONENTS)
        W, sigma2 = pcpca.gradient_descent_missing_data(fg, Y, n_iter=N_GD_ITER) #, learning_rate=LEARNING_RATE)

        X_imputed = pcpca.impute_missing_data(X)
        # X_imputed = (X_imputed.T + fg_mean).T
        imputation_mse = np.mean(
            (X_full[missing_mask_X] - X_imputed[missing_mask_X]) ** 2
        )
        print("PPCA {} missing, error: {}".format(missing_p, imputation_mse))
        imputation_errors_ppca[repeat_ii, ii] = imputation_mse



pcpca_results_df = pd.DataFrame(imputation_errors_pcpca, columns=missing_p_range - 0.015)
pcpca_results_df["method"] = ["PCPCA"] * pcpca_results_df.shape[0]

ppca_results_df = pd.DataFrame(imputation_errors_ppca, columns=missing_p_range - 0.005)
ppca_results_df["method"] = ["PPCA"] * ppca_results_df.shape[0]

samplemean_results_df = pd.DataFrame(imputation_errors_sample_means, columns=missing_p_range + 0.005)
samplemean_results_df["method"] = ["Sample means"] * samplemean_results_df.shape[0]

featuremean_results_df = pd.DataFrame(imputation_errors_feature_means, columns=missing_p_range + 0.015)
featuremean_results_df["method"] = ["Feature means"] * featuremean_results_df.shape[0]


results_df = pd.concat(
    [
        pcpca_results_df,
        ppca_results_df,
        samplemean_results_df,
        featuremean_results_df,
    ], axis=0
)

results_df.to_csv("./mouse_imputation_results.csv")

results_df_melted = pd.melt(results_df, id_vars="method")

plt.figure(figsize=(10.8, 5))
g = sns.lineplot(data=results_df_melted, x="variable", y="value", hue="method", err_style="bars")

plt.xlabel(r"Fraction missing")
plt.ylabel("MSE")
plt.title("Imputation (mouse data)")
plt.legend(bbox_to_anchor=(1.1, 1.05))
g.legend_.set_title(None)
plt.tight_layout()
plt.savefig("../../../plots/mouse_protein_expression/mouse_missing_data_imputation.png")
plt.show()
import ipdb; ipdb.set_trace()

plt.figure(figsize=(10.8, 5))
plt.errorbar(
    missing_p_range - 0.015,
    np.mean(imputation_errors_pcpca, axis=0),
    yerr=mean_confidence_interval(imputation_errors_pcpca),
    fmt="-o",
    label="PCPCA",
)
plt.errorbar(
    missing_p_range - 0.005,
    np.mean(imputation_errors_ppca, axis=0),
    yerr=mean_confidence_interval(imputation_errors_ppca),
    fmt="-o",
    label="PPCA",
)
plt.errorbar(
    missing_p_range + 0.005,
    np.mean(imputation_errors_sample_means, axis=0),
    yerr=mean_confidence_interval(imputation_errors_sample_means),
    fmt="-o",
    label="Sample means",
)
plt.errorbar(
    missing_p_range + 0.015,
    np.mean(imputation_errors_feature_means, axis=0),
    yerr=mean_confidence_interval(imputation_errors_feature_means),
    fmt="-o",
    label="Feature means",
)
plt.xticks(missing_p_range)
plt.xlabel(r"Fraction missing")
plt.ylabel("MSE")
plt.title("Imputation (mouse data)")
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()
plt.savefig("../../../plots/mouse_protein_expression/mouse_missing_data_imputation.png")
plt.show()


import ipdb; ipdb.set_trace()

# ipdb.set_trace()
