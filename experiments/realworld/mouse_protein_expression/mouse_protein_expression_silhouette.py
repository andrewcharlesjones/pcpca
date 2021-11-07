from pcpca import CPCA, PCPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import sys
from scipy.stats import ttest_ind

sys.path.append("../../../clvm")
from clvm import CLVM

DATA_PATH = "../../../data/mouse_protein_expression/clean/Data_Cortex_Nuclear.csv"
N_COMPONENTS = 2


if __name__ == "__main__":

    # Read in data
    data = pd.read_csv(DATA_PATH)
    data = data.fillna(0)

    # Get names of proteins
    protein_names = data.columns.values[1:78]

    data.Genotype[data.Genotype == "Control"] = "Non-DS"
    data.Genotype[data.Genotype == "Ts65Dn"] = "DS"

    # Background
    Y_df = data[
        (data.Behavior == "C/S")
        & (data.Genotype == "Non-DS")
        & (data.Treatment == "Saline")
    ]
    Y = Y_df[protein_names].values
    Y -= Y.mean(0)
    Y /= Y.std(0)
    Y = Y.T

    # Foreground
    X_df = data[(data.Behavior == "S/C") & (data.Treatment == "Saline")]
    # X_df = pd.concat([X_df.iloc[:177, :], X_df.iloc[180:, :]], axis=0)
    X = X_df[protein_names].values
    X -= X.mean(0)
    X /= X.std(0)
    X = X.T

    n, m = X.shape[1], Y.shape[1]

    import matplotlib

    font = {"size": 30}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    gamma_range_cpca = list(np.linspace(0, 400, 40))
    gamma_range_pcpca = list(np.linspace(0, 0.99, 40))
    # gamma_range_pcpca = [0, 0.5, 0.9]

    # print(X[:5, :])
    # import ipdb; ipdb.set_trace()

    cluster_scores_cpca = []
    cpca_gamma_plot_list = []
    for ii, gamma in enumerate(gamma_range_cpca):

        cpca = CPCA(gamma=n / m * gamma, n_components=N_COMPONENTS)
        X_reduced, Y_reduced = cpca.fit_transform(X, Y)

        X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

        try:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
        except:
            cpca_fail_gamma = gamma
            break
        cpca_gamma_plot_list.append(gamma)

        true_labels = pd.factorize(X_df.Genotype)[0]
        cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
        print("gamma'={}, cluster score={}".format(gamma, cluster_score))
        cluster_scores_cpca.append(cluster_score)

    cluster_scores_pcpca = []
    pcpca_gamma_plot_list = []
    for ii, gamma in enumerate(gamma_range_pcpca):

        # if gamma == 0.9:
        #     import ipdb; ipdb.set_trace()

        pcpca = PCPCA(gamma=n / m * gamma, n_components=N_COMPONENTS)
        X_reduced, Y_reduced = pcpca.fit_transform(X, Y)

        if pcpca.sigma2_mle <= 0:
            pcpca_fail_gamma = gamma
            break

        X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
        pcpca_gamma_plot_list.append(gamma)

        true_labels = pd.factorize(X_df.Genotype)[0]
        cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
        print("gamma'=*{}, cluster score={}".format(gamma, cluster_score))
        cluster_scores_pcpca.append(cluster_score)


    ## Fit CLVM
    # clvm = CLVM(
    #     data_dim=X.shape[0],
    #     n_bg=m,
    #     n_fg=n,
    #     latent_dim_shared=N_COMPONENTS,
    #     latent_dim_fg=N_COMPONENTS,
    # )
    # clvm.init_model()
    # clvm.fit_model(Y, X, n_iters=10000)
    # zy = clvm.qzy_mean.numpy().T
    # zx = clvm.qzx_mean.numpy().T
    # tx = clvm.qtx_mean.numpy().T
    # clvm_cluster_score = silhouette_score(X=tx, labels=true_labels)

    plt.figure(figsize=(38, 7))
    plt.subplot(151)
    plt.plot(cpca_gamma_plot_list, cluster_scores_cpca, "-o", linewidth=2)
    plt.title("CPCA")
    plt.ylim([0, 1])
    plt.xlim([0, cpca_gamma_plot_list[-1] + 40])
    plt.axvline(cpca_fail_gamma, color="black", linestyle="--")
    plt.axhline(np.max(cluster_scores_cpca), color="red", linestyle="--")
    # plt.axhline(clvm_cluster_score, color="blue", linestyle="--", label="CLVM")
    plt.xlabel(r"$\gamma^\prime$")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.subplot(152)
    plt.plot(pcpca_gamma_plot_list, cluster_scores_pcpca, "-o", linewidth=2)
    plt.title("PCPCA")
    plt.ylim([0, 1])
    plt.xlim([0, pcpca_gamma_plot_list[-1] + 0.1])
    plt.axvline(pcpca_fail_gamma, color="black", linestyle="--")
    plt.axhline(np.max(cluster_scores_pcpca), color="red", linestyle="--")
    # plt.axhline(clvm_cluster_score, color="blue", linestyle="--", label="CLVM")
    plt.xlabel(r"$\gamma^\prime$")
    plt.ylabel("Silhouette score")
    plt.legend()

    plt.subplot(153)
    cpca = CPCA(gamma=n / m * cpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
    X_reduced, Y_reduced = cpca.fit_transform(X, Y)

    plt.title(r"CPCA, $\gamma^\prime$={}".format(round(cpca_gamma_plot_list[-1], 2)))
    X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
    # [str(x) for x in kmeans.labels_]
    X_reduced_df["Genotype"] = X_df.Genotype.values

    Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
    Y_reduced_df["Genotype"] = ["Background" for _ in range(Y_reduced_df.shape[0])]

    results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
    results_df[["PCPC1", "PCPC2"]] = results_df[["PCPC1", "PCPC2"]] / results_df[
        ["PCPC1", "PCPC2"]
    ].std(0)

    g = sns.scatterplot(
        data=results_df,
        x="PCPC1",
        y="PCPC2",
        hue="Genotype",
        palette=["green", "orange", "gray"],
    )
    g.legend_.remove()
    plt.xlabel("CPC1")
    plt.ylabel("CPC2")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])

    plt.subplot(154)
    pcpca = PCPCA(gamma=n / m * pcpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
    X_reduced, Y_reduced = pcpca.fit_transform(X, Y)

    plt.title(r"PCPCA, $\gamma^\prime$={}".format(round(pcpca_gamma_plot_list[-1], 2)))
    X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
    X_reduced_df["Genotype"] = X_df.Genotype.values

    Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
    Y_reduced_df["Genotype"] = ["Background" for _ in range(Y_reduced_df.shape[0])]

    results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
    results_df[["PCPC1", "PCPC2"]] = results_df[["PCPC1", "PCPC2"]] / results_df[
        ["PCPC1", "PCPC2"]
    ].std(0)

    g = sns.scatterplot(
        data=results_df,
        x="PCPC1",
        y="PCPC2",
        hue="Genotype",
        palette=["green", "orange", "gray"],
    )
    g.legend_.remove()
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])

    ## Fit CLVM
    # plt.subplot(155)
    

    # zy_df = pd.DataFrame(zy, columns=["CLV1", "CLV2"])
    # zy_df["Genotype"] = ["Background"] * zy_df.shape[0]
    # tx_df = pd.DataFrame(tx, columns=["CLV1", "CLV2"])
    # tx_df["Genotype"] = X_df.Genotype.values
    
    # clvm_results_df = pd.concat([tx_df, zy_df], axis=0)
    # sns.scatterplot(
    #     data=clvm_results_df,
    #     x="CLV1",
    #     y="CLV2",
    #     hue="Genotype",
    #     palette=["green", "orange", "gray"],
    # )
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.title("CLVM")
    # ax = plt.gca()
    # handles, labels = ax.get_legend_handles_labels()

    plt.tight_layout()
    # plt.savefig("../../../plots/mouse_protein_expression/cluster_score_comparison.png")

    best_gamma_cpca = np.max(cluster_scores_cpca)
    best_gamma_pcpca = np.max(cluster_scores_pcpca)
    print(best_gamma_cpca)
    print(best_gamma_pcpca)
    # print(clvm_cluster_score)
    # plt.show()
    plt.close()

    ## Do bootstrap test to test significance of differences
    n_bootstraps = 100
    cpca_scores_bootstrap = []
    pcpca_scores_bootstrap = []
    n_X = X.shape[1]
    n_Y = Y.shape[1]
    for _ in range(n_bootstraps):
        subset_idx_X = np.random.choice(np.arange(n_X), size=int(n_X * 0.8), replace=False)
        subset_idx_Y = np.random.choice(np.arange(n_Y), size=int(n_Y * 0.8), replace=False)

        ## CPCA
        cpca = CPCA(gamma=n / m * best_gamma_cpca, n_components=N_COMPONENTS)
        X_reduced, Y_reduced = cpca.fit_transform(X[:, subset_idx_X], Y[:, subset_idx_Y])
        X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

        true_labels = pd.factorize(X_df.Genotype)[0][subset_idx_X]
        cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
        cpca_scores_bootstrap.append(cluster_score)

        ## PCPCA
        pcpca = PCPCA(gamma=n / m * best_gamma_pcpca, n_components=N_COMPONENTS)
        X_reduced, Y_reduced = pcpca.fit_transform(X[:, subset_idx_X], Y[:, subset_idx_Y])
        X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

        true_labels = pd.factorize(X_df.Genotype)[0][subset_idx_X]
        cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
        pcpca_scores_bootstrap.append(cluster_score)
    print(ttest_ind(pcpca_scores_bootstrap, cpca_scores_bootstrap))

    import ipdb

    ipdb.set_trace()









