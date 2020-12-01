import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import sys
from sklearn.decomposition import PCA
sys.path.append("../../../models")
from pcpca import PCPCA
from cpca import CPCA


DATA_PATH = "../../../data/mouse_protein_expression/clean/Data_Cortex_Nuclear.csv"
N_COMPONENTS = 2


if __name__ == "__main__":

	# Read in data
	data = pd.read_csv(DATA_PATH)
	data = data.fillna(0)

	# Get names of proteins
	protein_names = data.columns.values[1:78]

	# Background
	Y_df = data[(data.Behavior == "C/S") & (data.Genotype == "Control") & (data.Treatment == "Saline")]
	# Y_df = Y_df.iloc[:20, :]
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
	# import ipdb; ipdb.set_trace()

	n, m = X.shape[1], Y.shape[1]

	import matplotlib
	font = {'size'   : 15}
	matplotlib.rc('font', **font)

	# gamma_range_cpca = [0, 5, 10, 20, 40]
	# gamma_range_cpca.extend(np.arange(101, 400, 20))
	# gamma_range_cpca.extend(np.arange(401, 420))

	gamma_range_pcpca = list(np.linspace(0, 0.99, 5)) # [0, 0.5, 0.6, 0.7, 0.9]
	# gamma_range_pcpca.extend(np.arange(0.9, 1.3, 0.1))


	gamma_range_cpca = np.linspace(0, 20, 5)
	# gamma_range_pcpca = np.linspace(0, 1-1e-3, 10)


	sigma2_range = np.arange(0, 5, 0.5)
	# plt.figure(figsize=(14, 5))
	# ax_cpca = plt.gca()
	# plt.figure(figsize=(7, 5))
	# ax_pcpca = plt.gca()

	n_repeats = 10


	best_gammas_cpca = []
	best_gammas_pcpca = []

	best_rand_scores_cpca = np.empty((n_repeats, len(sigma2_range)))
	best_rand_scores_pcpca = np.empty((n_repeats, len(sigma2_range)))

	for repeat_ii in range(n_repeats):
		for sigma2_ii, sigma2 in enumerate(sigma2_range):

			print("Sigma2 = {}".format(sigma2))

			# Add noise
			curr_X = X + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=(p, n))
			curr_Y = Y + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=(p, m))

			rand_scores_cpca = []
			cpca_gamma_plot_list = []
			for ii, gamma in enumerate(gamma_range_cpca):

				# print(gamma)
				# gamma = gamma
				# pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
				# X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
				cpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
				X_reduced, Y_reduced = cpca.fit_transform(curr_X, curr_Y)

				# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
				# Y_reduced = (Y_reduced.T / Y_reduced.T.std(0)).T
				
				try:
					kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
				except:
					cpca_fail_gamma = gamma
					break
				cpca_gamma_plot_list.append(gamma)

				true_labels = pd.factorize(X_df.Genotype)[0]
				estimated_labels1 = kmeans.labels_
				estimated_labels2 = ~kmeans.labels_
				# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
				# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
				# rand_score = max(rand_score1, rand_score2)
				rand_score = silhouette_score(X=X_reduced.T, labels=true_labels)
				# print("gamma={}, rand score={}".format(gamma, rand_score))
				rand_scores_cpca.append(rand_score)

				# X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
				# X_reduced_df['Genotype'] = X_df.Genotype.values

				# Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
				# Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

				# results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

				# sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange','gray'])
				# plt.show()

			best_gamma = np.array(gamma_range_cpca)[np.argmax(np.array(rand_scores_cpca))]
			best_gammas_cpca.append(best_gamma)
			best_rand_scores_cpca[repeat_ii, sigma2_ii] = np.max(np.array(rand_scores_cpca))

			rand_scores_pcpca = []
			pcpca_gamma_plot_list = []
			for ii, gamma in enumerate(gamma_range_pcpca):
				# print(gamma)
				pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
				X_reduced, Y_reduced = pcpca.fit_transform(curr_X, curr_Y)

				if pcpca.sigma2_mle <= 0:
					pcpca_fail_gamma = gamma
					break

				X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
				Y_reduced = (Y_reduced.T / Y_reduced.T.std(0)).T
				kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
				pcpca_gamma_plot_list.append(gamma)

				true_labels = pd.factorize(X_df.Genotype)[0]
				estimated_labels1 = kmeans.labels_
				estimated_labels2 = ~kmeans.labels_
				# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
				# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
				# rand_score = max(rand_score1, rand_score2)
				rand_score = silhouette_score(X=X_reduced.T, labels=true_labels)
				# print("gamma=n/m*{}, rand score={}".format(gamma, rand_score))
				rand_scores_pcpca.append(rand_score)

				# X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
				# X_reduced_df['Genotype'] = X_df.Genotype.values

				# Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
				# Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

				# results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

				# sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange','gray'])
				# plt.show()

			best_gamma = np.array(gamma_range_pcpca)[np.argmax(np.array(rand_scores_pcpca))]
			best_gammas_pcpca.append(best_gamma)
			best_rand_scores_pcpca[repeat_ii, sigma2_ii] = np.max(np.array(rand_scores_pcpca))


	# import ipdb; ipdb.set_trace()
	plt.figure(figsize=(7, 5))
	# plt.plot(sigma2_range, best_rand_scores_cpca, '-o', label="CPCA")
	# plt.plot(sigma2_range, best_rand_scores_pcpca, '-o', label="PCPCA")

	plt.errorbar(sigma2_range, np.mean(best_rand_scores_cpca, axis=0), yerr=np.std(best_rand_scores_cpca, axis=0), fmt='-o', label="CPCA")
	plt.errorbar(sigma2_range, np.mean(best_rand_scores_pcpca, axis=0), yerr=np.std(best_rand_scores_pcpca, axis=0), fmt='-o', label="PCPCA")
	plt.legend()
	plt.xlabel("sigma2")
	plt.ylabel("Silhouette score")
	plt.tight_layout()
	plt.savefig("../../../plots/mouse_protein_expression/mouse_vary_sigma2.png")
	plt.show()



	
		

	# plt.savefig("../../plots/experiments/mouse_protein_expression/rand_index_comparison.png")
	# plt.show()


	# plt.savefig("../../plots/experiments/mouse_protein_expression/pcpca_vary_gamma.png")
	
	# plt.show()

	# import ipdb
	# ipdb.set_trace()
