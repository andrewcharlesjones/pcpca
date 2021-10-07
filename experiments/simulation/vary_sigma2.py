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

sys.path.append("../../models")
from pcpca import PCPCA
from cpca import CPCA


N_COMPONENTS = 1


if __name__ == "__main__":

    n, m = 200, 200
    p = 10

    # meanY = np.zeros(p)
    # mean = 0.5
    # meanX1, meanX2 = mean * np.ones(p), -mean * np.ones(p)
    # covY, covX = np.eye(p), np.eye(p)

    # Y = multivariate_normal.rvs(meanY, covY, size=m)
    # X1 = multivariate_normal.rvs(meanX1, covX, size=n//2)
    # X2 = multivariate_normal.rvs(meanX2, covX, size=n//2)
    # X = np.concatenate([X1, X2], axis=0)

    k = 1
    zy = np.random.normal(0, 1, size=(m, k))
    zx = np.random.normal(0, 1, size=(n, k))
    tx = np.random.normal(0, 1, size=(n, k))
    S = np.random.normal(0, 1, size=(k, p))
    W = np.random.normal(0, 10, size=(k, p))

    X = zx @ S + tx @ W + np.random.normal(size=(n, p))
    Y = zy @ S + np.random.normal(size=(m, p))

    X -= X.mean(0)
    X /= X.std(0)
    Y -= Y.mean(0)
    Y /= Y.std(0)

    X, Y = X.T, Y.T
    true_labels = np.zeros(n)
    true_labels[n // 2 :] = 1

    gamma_range_pcpca = list(np.linspace(0, 0.99, 10))
    gamma_range_cpca = np.linspace(0, 10, 10)

    sigma2_range = np.linspace(0, 10 ** 2, 10)

    best_gammas_cpca = []
    best_gammas_pcpca = []

    best_rand_scores_cpca = []
    best_rand_scores_pcpca = []

    best_w_corrs_cpca = []
    best_w_corrs_pcpca = []
    plt.figure(figsize=(14, 6))
    for sigma2 in sigma2_range:

        print("Sigma2 = {}".format(sigma2))

        # Add noise
        curr_X = X + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=(p, n))
        curr_Y = Y + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=(p, m))

        rand_scores_cpca = []
        w_corrs_cpca = []
        cpca_gamma_plot_list = []
        for ii, gamma in enumerate(gamma_range_cpca):

            # gamma = gamma
            # pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
            # X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
            cpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
            X_reduced, Y_reduced = cpca.fit_transform(curr_X, curr_Y)

            # import ipdb; ipdb.set_trace()

            X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

            try:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
            except:
                cpca_fail_gamma = gamma
                break
            cpca_gamma_plot_list.append(gamma)

            curr_W_corr = np.abs(pearsonr(cpca.W.squeeze(), W.squeeze())[0])
            w_corrs_cpca.append(curr_W_corr)
            # print("W correlation: {}".format(round(curr_W_corr, 2)))

            # true_labels = pd.factorize(X_df.Genotype)[0]
            estimated_labels1 = kmeans.labels_
            estimated_labels2 = ~kmeans.labels_
            rand_score1 = adjusted_rand_score(
                labels_true=true_labels, labels_pred=estimated_labels1
            )
            rand_score2 = adjusted_rand_score(
                labels_true=true_labels, labels_pred=estimated_labels2
            )
            rand_score = max(rand_score1, rand_score2)
            # print("gamma={}, rand score={}".format(gamma, rand_score))
            rand_scores_cpca.append(rand_score)

        # import ipdb; ipdb.set_trace()
        best_gamma = np.array(gamma_range_cpca)[np.argmax(np.array(rand_scores_cpca))]
        best_gammas_cpca.append(best_gamma)
        best_rand_scores_cpca.append(np.max(np.array(rand_scores_cpca)))
        best_w_corrs_cpca.append(np.max(np.array(w_corrs_cpca)))

        rand_scores_pcpca = []
        w_corrs_pcpca = []
        pcpca_gamma_plot_list = []
        for ii, gamma in enumerate(gamma_range_pcpca):

            pcpca = PCPCA(gamma=n / m * gamma, n_components=N_COMPONENTS)
            X_reduced, Y_reduced = pcpca.fit_transform(curr_X, curr_Y)

            if pcpca.sigma2_mle <= 0:
                pcpca_fail_gamma = gamma
                break

            X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

            curr_W_corr = np.abs(pearsonr(pcpca.W_mle.squeeze(), W.squeeze())[0])
            w_corrs_pcpca.append(curr_W_corr)

            kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
            pcpca_gamma_plot_list.append(gamma)

            estimated_labels1 = kmeans.labels_
            estimated_labels2 = ~kmeans.labels_
            rand_score1 = adjusted_rand_score(
                labels_true=true_labels, labels_pred=estimated_labels1
            )
            rand_score2 = adjusted_rand_score(
                labels_true=true_labels, labels_pred=estimated_labels2
            )
            rand_score = max(rand_score1, rand_score2)
            # print("gamma=n/m*{}, rand score={}".format(gamma, rand_score))
            rand_scores_pcpca.append(rand_score)

        best_gamma = np.array(gamma_range_pcpca)[np.argmax(np.array(rand_scores_pcpca))]
        best_gammas_pcpca.append(best_gamma)
        best_rand_scores_pcpca.append(np.max(np.array(rand_scores_pcpca)))
        best_w_corrs_pcpca.append(np.max(np.array(w_corrs_pcpca)))

    plt.plot(sigma2_range, best_w_corrs_cpca, label="CPCA")
    plt.plot(sigma2_range, best_w_corrs_pcpca, label="PCPCA")
    plt.legend()
    plt.show()

    # 	plt.subplot(121)
    # 	# plt.sca(ax_cpca)
    # 	plt.plot(cpca_gamma_plot_list, rand_scores_cpca, '-o', linewidth=2, label="CPCA, sigma2={}".format(sigma2))
    # 	plt.title("CPCA")
    # 	plt.ylim([0, 1])
    # 	# plt.axvline(cpca_fail_gamma, color="black", linestyle="--")
    # 	plt.xlabel("gamma")
    # 	plt.ylabel("Adjusted Rand index")
    # 	plt.subplot(122)
    # 	# plt.sca(ax_pcpca)
    # 	print("sigma2={}".format(sigma2))
    # 	plt.plot(pcpca_gamma_plot_list, rand_scores_pcpca, '-o', linewidth=2, label="{}".format(sigma2))
    # 	plt.title("PCPCA")
    # 	plt.ylim([0, 1])
    # 	# plt.axvline(pcpca_fail_gamma, color="black", linestyle="--")
    # 	plt.xlabel("gamma * m/n")
    # 	plt.ylabel("Adjusted Rand index")
    # 	plt.legend()
    # 	plt.legend(title='sigma2', bbox_to_anchor=(1.05, 1), loc='upper left')

    # plt.show()

    # plt.figure(figsize=(7, 5))
    # # plt.plot(sigma2_range, best_rand_scores_cpca, label="CPCA")
    # # plt.plot(sigma2_range, best_rand_scores_pcpca, label="PCPCA")
    # plt.plot(sigma2_range, best_gammas_cpca, label="CPCA")
    # plt.plot(sigma2_range, best_gammas_pcpca, label="PCPCA")
    # plt.legend()
    # plt.xlabel("sigma2")
    # plt.ylabel("Best gamma")
    # plt.show()
