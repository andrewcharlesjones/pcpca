import ipdb
import matplotlib
from pcpca import PCPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
import sys
from sklearn.decomposition import PCA
from numpy.linalg import slogdet

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

inv = np.linalg.inv

DATA_PATH = "../../../data/mouse_protein_expression/clean/Data_Cortex_Nuclear.csv"
N_COMPONENTS = 2

# Read in data
data = pd.read_csv(DATA_PATH)


# Separate into background and foreground data
# In this case,
# background data is data from mice who did not receive shock therapty
# foreground data is from mice who did receive shock therapy

# Fill NAs
data = data.fillna(0)

# Get names of proteins
protein_names = data.columns.values[1:78]

# Background
Y_df = data[
    (data.Behavior == "C/S")
    & (data.Genotype == "Control")
    & (data.Treatment == "Saline")
]
Y = Y_df[protein_names].values
Y -= Y.mean(0)
Y /= Y.std(0)
Y_full = Y.T


# Foreground
X_df = data[(data.Behavior == "S/C") & (data.Treatment == "Saline")]
X = X_df[protein_names].values
X -= X.mean(0)
X /= X.std(0)
X_full = X.T

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


gamma = 0.4
missing_p_range = [0.0, 0.3, 0.5, 0.7]
n_repeats = 1
W_errors = np.empty((n_repeats, len(missing_p_range)))

plt.figure(figsize=(7 * len(missing_p_range), 6))

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

    pcpca = PCPCA(gamma=gamma, n_components=N_COMPONENTS)
    W, sigma2 = pcpca.gradient_descent_missing_data(X, Y, n_iter=200)
    pcpca.W_mle = W
    pcpca.sigma2_mle = sigma2

    X_reduced, Y_reduced = pcpca.transform(X_full, Y_full)
    X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

    true_labels = pd.factorize(X_df.Genotype)[0]
    cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
    print("gamma=n/m*{}, cluster score={}".format(gamma, cluster_score))

    # Plot reduced foreground data
    X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
    X_reduced_df["Genotype"] = X_df.Genotype.values

    Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
    Y_reduced_df["Genotype"] = ["Background" for _ in range(Y_reduced_df.shape[0])]

    plt.subplot(1, len(missing_p_range), ii + 1)
    sns.scatterplot(
        data=X_reduced_df,
        x="PCPC1",
        y="PCPC2",
        hue="Genotype",
        palette=["green", "orange"],
    )

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])

    ax.legend(prop={"size": 20})

    plt.xlabel(r"PCPC1")
    plt.ylabel("PCPC2")
    plt.title(
        "Fraction missing: {}\nSilhouette score: {}".format(
            missing_p, round(cluster_score, 2)
        )
    )

plt.tight_layout()
plt.savefig("../../../plots/mouse_protein_expression/mouse_missing_data.png")
plt.show()


ipdb.set_trace()
