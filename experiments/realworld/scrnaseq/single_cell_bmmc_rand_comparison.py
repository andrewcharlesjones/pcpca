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
import sys
sys.path.append("../../../models")
from pcpca import PCPCA
from cpca import CPCA


DATA_DIR = "../../../data/singlecell_bmmc"
N_COMPONENTS = 5


if __name__ == "__main__":

	# Read in data
	# pretransplant1 = pd.read_csv(pjoin(DATA_DIR, "clean", "pretransplant1.csv"), index_col=0)
	# posttransplant1 = pd.read_csv(pjoin(DATA_DIR, "clean", "posttransplant1.csv"), index_col=0)

	pretransplant2 = pd.read_csv(pjoin(DATA_DIR, "clean", "pretransplant2.csv"), index_col=0)
	posttransplant2 = pd.read_csv(pjoin(DATA_DIR, "clean", "posttransplant2.csv"), index_col=0)

	healthy1 = pd.read_csv(pjoin(DATA_DIR, "clean", "healthy1.csv"), index_col=0)
	healthy2 = pd.read_csv(pjoin(DATA_DIR, "clean", "healthy2.csv"), index_col=0)



	## Background is made up of healthy cells
	Y = healthy1.values # pd.concat([healthy1, healthy2], axis=0).values

	## Foreground is made up of AML cells
	# X = pd.concat([pretransplant1, pretransplant2, posttransplant1, posttransplant2], axis=0).values
	# X_labels = ["Pretransplant1" for _ in range(pretransplant1.shape[0])]
	# X_labels.extend(["Pretransplant2" for _ in range(pretransplant2.shape[0])])
	# X_labels.extend(["Posttransplant1" for _ in range(posttransplant1.shape[0])])
	# X_labels.extend(["Posttransplant2" for _ in range(posttransplant2.shape[0])])

	X = pd.concat([pretransplant2, posttransplant2], axis=0).values
	X_labels = ["Pretransplant2" for _ in range(pretransplant2.shape[0])]
	X_labels.extend(["Posttransplant2" for _ in range(posttransplant2.shape[0])])
	X_labels = np.array(X_labels)
	assert X_labels.shape[0] == X.shape[0]

	# import ipdb; ipdb.set_trace()
	

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

	# plt.subplot(221)
	# out = PCA().fit_transform(pretransplant2)
	# plt.scatter(out[:, 0], out[:, 1], label="Pretransplat")
	# plt.subplot(222)
	# out = PCA().fit_transform(posttransplant2)
	# plt.scatter(out[:, 0], out[:, 1], label="Posttransplat")
	# plt.subplot(223)
	# out = PCA().fit_transform(healthy1)
	# plt.scatter(out[:, 0], out[:, 1], label="healthy1")
	# plt.subplot(224)
	# out = PCA().fit_transform(healthy2)
	# plt.scatter(out[:, 0], out[:, 1], label="healthy2")
	# plt.legend()
	# plt.show()

	# X1 = PCA(n_components=500).fit_transform(pretransplant2)
	# X2 = PCA(n_components=500).fit_transform(posttransplant2)
	# Y1 = PCA(n_components=500).fit_transform(healthy1)
	# Y2 = PCA(n_components=500).fit_transform(healthy2)

	# X = np.concatenate([X1, X2], axis=0).T
	# Y = np.concatenate([Y1, Y2], axis=0).T

	# sys.exit()

	import matplotlib
	font = {'size'   : 15}
	matplotlib.rc('font', **font)

	# gamma_range_cpca = np.linspace(10, 15, 30)
	gamma_range_cpca = np.linspace(0, 10, 20)
	# gamma_range_cpca = [0, 6.0]

	rand_scores_cpca = []
	cpca_gamma_plot_list = []
	for ii, gamma in enumerate(gamma_range_cpca):

		cpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
		X_reduced, Y_reduced = cpca.fit_transform(X, Y)
		X_reduced = X_reduced[1:3, :]
		
		try:
			kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
		except:
			cpca_fail_gamma = gamma
			break
		cpca_gamma_plot_list.append(gamma)

		true_labels = pd.factorize(X_df.condition)[0]
		# estimated_labels1 = kmeans.labels_
		# estimated_labels2 = 1-kmeans.labels_
		# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
		# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
		# rand_score = max(rand_score1, rand_score2)
		rand_score = silhouette_score(X=X_reduced.T, labels=true_labels)
		print("gamma={}, rand score={}".format(gamma, rand_score))
		rand_scores_cpca.append(rand_score)


		X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
		X_reduced_df['condition'] = X_labels
		plot_df = X_reduced_df[X_reduced_df.condition.isin(["Pretransplant2", "Posttransplant2"])]

		# plt.subplot(121)
		# sns.scatterplot(data=plot_df, x="PCPC1", y="PCPC2", hue=estimated_labels2, alpha=0.1)
		# plt.subplot(122)
		# plt.subplot(1, 3, ii+1)
		# sns.scatterplot(data=plot_df, x="PCPC1", y="PCPC2", hue="condition", alpha=0.1)
		# plt.title("gamma={}".format(round(gamma, 2)))
		# plt.xlabel("CPC1")
		# plt.ylabel("CPC2")
		# plt.show()

	gamma_range_pcpca = np.linspace(0, 1-1e-3, 20)

	rand_scores_pcpca = []
	pcpca_gamma_plot_list = []
	for ii, gamma in enumerate(gamma_range_pcpca):

		pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
		X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
		X_reduced = X_reduced[2:4, :]

		if pcpca.sigma2_mle <= 0:
			pcpca_fail_gamma = gamma
			break
		
		# try:
		# 	kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
		# except:
		# 	cpca_fail_gamma = gamma
		# 	break
		pcpca_gamma_plot_list.append(gamma)

		true_labels = pd.factorize(X_df.condition)[0]
		# estimated_labels1 = kmeans.labels_
		# estimated_labels2 = 1-kmeans.labels_
		# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
		# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
		# rand_score = max(rand_score1, rand_score2)
		rand_score = silhouette_score(X=X_reduced.T, labels=true_labels)
		print("gamma={}, rand score={}".format(gamma, rand_score))
		rand_scores_pcpca.append(rand_score)


		X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
		X_reduced_df['condition'] = X_labels
		plot_df = X_reduced_df[X_reduced_df.condition.isin(["Pretransplant2", "Posttransplant2"])]

		# plt.subplot(121)
		# sns.scatterplot(data=plot_df, x="PCPC1", y="PCPC2", hue=estimated_labels2, alpha=0.1)
		# plt.subplot(122)
		# plt.subplot(1, 3, 3)
		# sns.scatterplot(data=plot_df, x="PCPC1", y="PCPC2", hue="condition", alpha=0.1)
		# plt.xlabel("CPC2")
		# plt.ylabel("CPC3")
		# plt.title("gamma={}".format(round(gamma, 2)))
		# plt.show()
		
	# plt.plot(cpca_gamma_plot_list, rand_scores_cpca, '-o', linewidth=2)
	# plt.show()
	# sns.scatterplot(data=plot_df, x="PCPC1", y="PCPC2", hue="condition", alpha=0.1)

	plt.figure(figsize=(14, 6))
	plt.subplot(121)
	plt.plot(cpca_gamma_plot_list, rand_scores_cpca, '-o', linewidth=2)
	plt.title("CPCA")
	plt.ylim([0, 1])
	plt.axvline(cpca_fail_gamma, color="black", linestyle="--")
	plt.xlabel("gamma")
	plt.ylabel("Silhouette score")
	plt.subplot(122)
	plt.plot(pcpca_gamma_plot_list, rand_scores_pcpca, '-o', linewidth=2)
	plt.title("PCPCA")
	plt.ylim([0, 1])
	plt.axvline(pcpca_fail_gamma, color="black", linestyle="--")
	plt.xlabel("gamma * m/n")
	plt.ylabel("Silhouette score")
	plt.savefig("../../../plots/scrnaseq/singlecell_silhouette_score.png")
	plt.show()
	# plt.tight_layout()

	


	plt.subplot(223)
	cpca = CPCA(gamma=cpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
	X_reduced, Y_reduced = cpca.fit_transform(X, Y)
	X_reduced, Y_reduced = X_reduced[1:3, :], Y_reduced[1:3, :]
	# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

	plt.title("CPCA, gamma={}".format(round(cpca_gamma_plot_list[-1], 2)))
	X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
	X_reduced_df['condition'] = X_labels

	Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
	Y_reduced_df['condition'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

	results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
	results_df[["PCPC1", "PCPC2"]] = results_df[["PCPC1", "PCPC2"]] / results_df[["PCPC1", "PCPC2"]].std(0)

	sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="condition", palette=['green','orange', 'gray'], alpha=0.5)
	plt.xlabel("CPC1")
	plt.ylabel("CPC2")


	plt.subplot(224)
	pcpca = PCPCA(gamma=n/m*pcpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
	X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
	X_reduced, Y_reduced = X_reduced[2:4, :], Y_reduced[2:4, :]
	# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

	plt.title("PCPCA, gamma*m/n={}".format(round(pcpca_gamma_plot_list[-1], 2)))
	X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
	X_reduced_df['condition'] = X_labels

	Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
	Y_reduced_df['condition'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

	results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
	results_df[["PCPC1", "PCPC2"]] = results_df[["PCPC1", "PCPC2"]] / results_df[["PCPC1", "PCPC2"]].std(0)
	# results_df = results_df / results_df.std(0)

	sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="condition", palette=['green','orange', 'gray'], alpha=0.5)


	plt.show()
	import ipdb; ipdb.set_trace()

	# import ipdb
	# ipdb.set_trace()
