from pcpca import PCPCA, CPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from scipy import stats
from tqdm import tqdm

DATA_PATH = "../../../data/mouse_protein_expression/clean/Data_Cortex_Nuclear.csv"
N_COMPONENTS = 2


def mean_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    m, se = np.mean(data, axis=0), stats.sem(data, axis=0)
    width = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return width


if __name__ == "__main__":

    # Read in data
    data = pd.read_csv(DATA_PATH)
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
    Y = Y.T

    # Foreground
    X_df = data[(data.Behavior == "S/C") & (data.Treatment == "Saline")]
    X_df = pd.concat([X_df.iloc[:177, :], X_df.iloc[180:, :]], axis=0)
    X = X_df[protein_names].values
    X -= X.mean(0)
    X /= X.std(0)
    X = X.T

    p = X.shape[0]

    n, m = X.shape[1], Y.shape[1]

    import matplotlib

    font = {"size": 20}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    gamma_range_pcpca = list(np.linspace(0, 0.99, 5))
    gamma_range_cpca = np.linspace(0, 20, 5)

    sigma2_range = np.arange(0, 5.5, 0.5)

    n_repeats = 100

    best_gammas_cpca = []
    best_gammas_pcpca = []

    best_cluster_scores_cpca = np.empty((n_repeats, len(sigma2_range)))
    best_cluster_scores_pcpca = np.empty((n_repeats, len(sigma2_range)))

    with tqdm(total=n_repeats * len(sigma2_range)) as pbar:
        for repeat_ii in range(n_repeats):
            for sigma2_ii, sigma2 in enumerate(sigma2_range):

                # print("Sigma2 = {}".format(sigma2))

                # Add noise
                curr_X = X + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=(p, n))
                curr_Y = Y + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=(p, m))

                cluster_scores_cpca = []
                cpca_gamma_plot_list = []
                for ii, gamma in enumerate(gamma_range_cpca):

                    cpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
                    X_reduced, Y_reduced = cpca.fit_transform(curr_X, curr_Y)

                    try:
                        kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
                    except:
                        cpca_fail_gamma = gamma
                        break
                    cpca_gamma_plot_list.append(gamma)

                    true_labels = pd.factorize(X_df.Genotype)[0]
                    cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
                    cluster_scores_cpca.append(cluster_score)

                best_gamma = np.array(gamma_range_cpca)[
                    np.argmax(np.array(cluster_scores_cpca))
                ]
                best_gammas_cpca.append(best_gamma)
                best_cluster_scores_cpca[repeat_ii, sigma2_ii] = np.max(
                    np.array(cluster_scores_cpca)
                )

                cluster_scores_pcpca = []
                pcpca_gamma_plot_list = []
                for ii, gamma in enumerate(gamma_range_pcpca):

                    pcpca = PCPCA(gamma=n / m * gamma, n_components=N_COMPONENTS)
                    X_reduced, Y_reduced = pcpca.fit_transform(curr_X, curr_Y)

                    if pcpca.sigma2_mle <= 0:
                        pcpca_fail_gamma = gamma
                        break

                    X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
                    Y_reduced = (Y_reduced.T / Y_reduced.T.std(0)).T
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
                    pcpca_gamma_plot_list.append(gamma)

                    true_labels = pd.factorize(X_df.Genotype)[0]
                    cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
                    cluster_scores_pcpca.append(cluster_score)

                best_gamma = np.array(gamma_range_pcpca)[
                    np.argmax(np.array(cluster_scores_pcpca))
                ]
                best_gammas_pcpca.append(best_gamma)
                best_cluster_scores_pcpca[repeat_ii, sigma2_ii] = np.max(
                    np.array(cluster_scores_pcpca)
                )

                pbar.update(1)

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        sigma2_range,
        np.mean(best_cluster_scores_pcpca, axis=0),
        yerr=mean_confidence_interval(best_cluster_scores_pcpca),
        fmt="-o",
        label="PCPCA",
    )
    plt.errorbar(
        sigma2_range,
        np.mean(best_cluster_scores_cpca, axis=0),
        yerr=mean_confidence_interval(best_cluster_scores_cpca),
        fmt="-o",
        label="CPCA",
    )
    plt.legend()
    plt.xlabel(r"$\sigma^2$")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig("../../../plots/mouse_protein_expression/mouse_vary_sigma2.png")
    plt.show()
