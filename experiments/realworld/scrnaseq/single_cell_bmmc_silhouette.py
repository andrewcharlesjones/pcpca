from pcpca import PCPCA, CPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
from scipy.io import mmread
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score



DATA_DIR = "../../../data/singlecell_bmmc"
N_COMPONENTS = 5


if __name__ == "__main__":

    # Read in data
    # pretransplant1 = pd.read_csv(pjoin(DATA_DIR, "clean", "pretransplant1.csv"), index_col=0)
    # posttransplant1 = pd.read_csv(pjoin(DATA_DIR, "clean", "posttransplant1.csv"), index_col=0)

    pretransplant2 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "pretransplant2.csv"), index_col=0)
    posttransplant2 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "posttransplant2.csv"), index_col=0)

    healthy1 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "healthy1.csv"), index_col=0)
    # healthy2 = pd.read_csv(pjoin(DATA_DIR, "clean", "healthy2.csv"), index_col=0)

    # Background is made up of healthy cells
    Y = healthy1.values  # pd.concat([healthy1, healthy2], axis=0).values

    X = pd.concat([pretransplant2, posttransplant2], axis=0).values
    X_labels = ["Pretransplant2" for _ in range(pretransplant2.shape[0])]
    X_labels.extend(
        ["Posttransplant2" for _ in range(posttransplant2.shape[0])])
    X_labels = np.array(X_labels)
    assert X_labels.shape[0] == X.shape[0]

    # Standardize
    Y -= Y.mean(0)
    Y /= Y.std(0)
    Y = Y.T
    X -= X.mean(0)
    X /= X.std(0)
    X = X.T

    n, m = X.shape[1], Y.shape[1]

    X_df = pd.DataFrame(X.T)
    X_df['condition'] = X_labels

    import matplotlib
    font = {'size': 20}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['text.usetex'] = True

    gamma_range_cpca = np.linspace(0, 5, 20)

    cluster_scores_cpca = []
    cpca_gamma_plot_list = []
    for ii, gamma in enumerate(gamma_range_cpca):

        cpca = CPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
        X_reduced, Y_reduced = cpca.fit_transform(X, Y)
        X_reduced = X_reduced[1:3, :]

        try:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
        except:
            cpca_fail_gamma = gamma
            break
        cpca_gamma_plot_list.append(gamma)

        true_labels = pd.factorize(X_df.condition)[0]
        cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
        print("gamma={}, cluster score={}".format(gamma, cluster_score))
        cluster_scores_cpca.append(cluster_score)

        X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
        X_reduced_df['condition'] = X_labels
        plot_df = X_reduced_df[X_reduced_df.condition.isin(
            ["Pretransplant2", "Posttransplant2"])]

    gamma_range_pcpca = np.linspace(0, 1-1e-3, 20)

    cluster_scores_pcpca = []
    pcpca_gamma_plot_list = []
    for ii, gamma in enumerate(gamma_range_pcpca):

        pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
        X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
        X_reduced = X_reduced[2:4, :]

        if pcpca.sigma2_mle <= 0:
            pcpca_fail_gamma = gamma
            break

        pcpca_gamma_plot_list.append(gamma)

        true_labels = pd.factorize(X_df.condition)[0]
        cluster_score = silhouette_score(X=X_reduced.T, labels=true_labels)
        print("gamma={}, cluster score={}".format(gamma, cluster_score))
        cluster_scores_pcpca.append(cluster_score)

        X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
        X_reduced_df['condition'] = X_labels
        plot_df = X_reduced_df[X_reduced_df.condition.isin(
            ["Pretransplant2", "Posttransplant2"])]

    plt.figure(figsize=(14, 6))

    plt.subplot(121)
    plt.plot(cpca_gamma_plot_list, cluster_scores_cpca, '-o', linewidth=2)
    plt.title("CPCA")
    plt.ylim([0, 1])
    plt.axvline(cpca_fail_gamma, color="black", linestyle="--")
    plt.axhline(np.max(cluster_scores_cpca), color="red", linestyle="--")
    plt.xlabel(r'$\gamma^\prime$')
    plt.ylabel("Silhouette score")

    plt.subplot(122)
    plt.plot(pcpca_gamma_plot_list, cluster_scores_pcpca, '-o', linewidth=2)
    plt.title("PCPCA")
    plt.ylim([0, 1])
    plt.axvline(pcpca_fail_gamma, color="black", linestyle="--")
    plt.axhline(np.max(cluster_scores_pcpca), color="red", linestyle="--")
    plt.xlabel(r'$\gamma^\prime$')
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig("../../../plots/scrnaseq/singlecell_silhouette_score.png")
    plt.show()

    # plt.subplot(223)
    # cpca = CPCA(gamma=cpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
    # X_reduced, Y_reduced = cpca.fit_transform(X, Y)
    # X_reduced, Y_reduced = X_reduced[1:3, :], Y_reduced[1:3, :]

    # plt.title("CPCA, gamma={}".format(round(cpca_gamma_plot_list[-1], 2)))
    # X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
    # X_reduced_df['condition'] = X_labels

    # Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
    # Y_reduced_df['condition'] = [
    #     "Background" for _ in range(Y_reduced_df.shape[0])]

    # results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
    # results_df[["PCPC1", "PCPC2"]] = results_df[[
    #     "PCPC1", "PCPC2"]] / results_df[["PCPC1", "PCPC2"]].std(0)

    # sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="condition", palette=[
    #                 'green', 'orange', 'gray'], alpha=0.5)
    # plt.xlabel("CPC1")
    # plt.ylabel("CPC2")

    # plt.subplot(224)
    # pcpca = PCPCA(
    #     gamma=n/m*pcpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
    # X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
    # X_reduced, Y_reduced = X_reduced[2:4, :], Y_reduced[2:4, :]

    # plt.title(
    #     "PCPCA, gamma*m/n={}".format(round(pcpca_gamma_plot_list[-1], 2)))
    # X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
    # X_reduced_df['condition'] = X_labels

    # Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
    # Y_reduced_df['condition'] = [
    #     "Background" for _ in range(Y_reduced_df.shape[0])]

    # results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
    # results_df[["PCPC1", "PCPC2"]] = results_df[[
    #     "PCPC1", "PCPC2"]] / results_df[["PCPC1", "PCPC2"]].std(0)

    # sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="condition", palette=[
    #                 'green', 'orange', 'gray'], alpha=0.5)

    # plt.show()
    import ipdb
    ipdb.set_trace()
