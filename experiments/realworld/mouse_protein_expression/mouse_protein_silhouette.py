import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
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

	n, m = X.shape[1], Y.shape[1]

	import matplotlib
	font = {'size'   : 15}
	matplotlib.rc('font', **font)

	gamma_range_cpca = [0, 5, 10, 20, 40]
	gamma_range_cpca.extend(np.linspace(101, 600, 40))
	gamma_range_pcpca = list(np.linspace(0, 0.99, 40)) # [0, 0.5, 0.6, 0.7, 0.9]
	gamma_range_pcpca.extend(np.arange(0.9, 1.3, 0.03))

	rand_scores_cpca = []
	cpca_gamma_plot_list = []
	for ii, gamma in enumerate(gamma_range_cpca):
		# gamma = gamma
		# pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
		# X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
		cpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
		X_reduced, Y_reduced = cpca.fit_transform(X, Y)

		X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
		
		try:
			kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
		except:
			cpca_fail_gamma = gamma
			break
		cpca_gamma_plot_list.append(gamma)

		true_labels = pd.factorize(X_df.Genotype)[0]
		# estimated_labels1 = kmeans.labels_
		# estimated_labels2 = ~kmeans.labels_
		# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
		# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
		# rand_score = max(rand_score1, rand_score2)
		rand_score = silhouette_score(X=X_reduced.T, labels=true_labels)
		print("gamma={}, rand score={}".format(gamma, rand_score))
		rand_scores_cpca.append(rand_score)

	rand_scores_pcpca = []
	pcpca_gamma_plot_list = []
	for ii, gamma in enumerate(gamma_range_pcpca):
		gamma = gamma
		pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
		X_reduced, Y_reduced = pcpca.fit_transform(X, Y)

		if pcpca.sigma2_mle <= 0:
			pcpca_fail_gamma = gamma
			break

		X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
		kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
		pcpca_gamma_plot_list.append(gamma)

		true_labels = pd.factorize(X_df.Genotype)[0]
		# estimated_labels1 = kmeans.labels_
		# estimated_labels2 = ~kmeans.labels_
		# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
		# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
		# rand_score = max(rand_score1, rand_score2)
		rand_score = silhouette_score(X=X_reduced.T, labels=true_labels)
		print("gamma=n/m*{}, rand score={}".format(gamma, rand_score))
		rand_scores_pcpca.append(rand_score)


		# pcpca = PCPCA(gamma=gamma, n_components=N_COMPONENTS)
		# X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
		
		# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
		# kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)

		# true_labels = pd.factorize(X_df.Genotype)[0]
		# estimated_labels1 = kmeans.labels_
		# estimated_labels2 = ~kmeans.labels_
		# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
		# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
		# rand_score = max(rand_score1, rand_score2)
		# print("gamma={}, rand score={}".format(gamma, rand_score))
		# rand_scores_cpca.append(rand_score)
		# import ipdb; ipdb.set_trace()


	# 	plt.subplot(1, len(gamma_range), ii+1)
		# if gamma == 0:
		# 	plt.title("gamma={}*n/m (PPCA)\nsigma2={}".format(gamma, round(pcpca.sigma2_mle, 2)))
		# else:
		# 	plt.title("gamma={}*n/m\nsigma2={}".format(gamma, round(pcpca.sigma2_mle, 2)))

		# Plot reduced foreground data
		# X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
		# X_reduced_df['Genotype'] = X_df.Genotype.values # [str(x) for x in kmeans.labels_]

		# Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
		# Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

		# # results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

		# sns.scatterplot(data=X_reduced_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange'])
		# plt.xlabel("CPC1")
		# plt.ylabel("CPC2")
		# plt.show()


	plt.figure(figsize=(28, 6))
	plt.subplot(141)
	plt.plot(cpca_gamma_plot_list, rand_scores_cpca, '-o', linewidth=2)
	plt.title("CPCA")
	plt.ylim([0, 1])
	plt.axvline(cpca_fail_gamma, color="black", linestyle="--")
	plt.xlabel("gamma")
	plt.ylabel("Adjusted Rand index")
	plt.subplot(142)
	plt.plot(pcpca_gamma_plot_list, rand_scores_pcpca, '-o', linewidth=2)
	plt.title("PCPCA")
	plt.ylim([0, 1])
	plt.axvline(pcpca_fail_gamma, color="black", linestyle="--")
	plt.xlabel("gamma * m/n")
	plt.ylabel("Adjusted Rand index")
	# plt.tight_layout()

	


	plt.subplot(143)
	cpca = CPCA(gamma=cpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
	X_reduced, Y_reduced = cpca.fit_transform(X, Y)
	# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

	plt.title("CPCA, gamma={}".format(round(cpca_gamma_plot_list[-1], 2)))
	X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
	X_reduced_df['Genotype'] = X_df.Genotype.values # [str(x) for x in kmeans.labels_]

	Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
	Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

	results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
	results_df[["PCPC1", "PCPC2"]] = results_df[["PCPC1", "PCPC2"]] / results_df[["PCPC1", "PCPC2"]].std(0)

	sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange', 'gray'])
	plt.xlabel("CPC1")
	plt.ylabel("CPC2")


	plt.subplot(144)
	pcpca = PCPCA(gamma=n/m*pcpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
	X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
	# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

	plt.title("PCPCA, gamma*m/n={}".format(round(pcpca_gamma_plot_list[-1], 2)))
	X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
	X_reduced_df['Genotype'] = X_df.Genotype.values # [str(x) for x in kmeans.labels_]

	Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
	Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

	results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
	results_df[["PCPC1", "PCPC2"]] = results_df[["PCPC1", "PCPC2"]] / results_df[["PCPC1", "PCPC2"]].std(0)
	# results_df = results_df / results_df.std(0)

	sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange', 'gray'])


	plt.savefig("../../../plots/mouse_protein_expression/rand_index_comparison.png")
	plt.tight_layout()
	plt.show()


	# plt.savefig("../../plots/experiments/mouse_protein_expression/pcpca_vary_gamma.png")
	
	# plt.show()

	# import ipdb
	# ipdb.set_trace()
