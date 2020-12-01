import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
from scipy.io import mmread
from sklearn.decomposition import PCA
import sys
sys.path.append("../../../models")
from pcpca import PCPCA
from cpca import CPCA


DATA_DIR = "../../../data/singlecell_bmmc"
N_COMPONENTS = 10


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
	X_labels = ["Pretransplant" for _ in range(pretransplant2.shape[0])]
	X_labels.extend(["Posttransplant" for _ in range(posttransplant2.shape[0])])
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

	gamma_range = [0, 0.7, 0.9]
	# gamma_range = np.linspace(0.9, 2, 3, 5)
	# gamma_range = [0, 0.8, 0.9, 0.99] #, 1.5, 2.0, 3.0, 5.0]
	plt.figure(figsize=(len(gamma_range) * 7, 5))

	for ii, gamma in enumerate(gamma_range):
		# gamma = n/m*gamma

		# cpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
		# X_reduced, Y_reduced = cpca.fit_transform(X, Y)
		pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
		X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
		

		plt.subplot(1, len(gamma_range), ii+1)
		if gamma == 0:
			plt.title("Gamma = {} (PPCA)".format(round(gamma, 3)))
		else:
			plt.title("Gamma = {}".format(round(gamma, 3)))

		X_reduced_df = pd.DataFrame(X_reduced.T[:, 2:4], columns=["PCPC1", "PCPC2"])
		X_reduced_df['condition'] = X_labels

		# Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
		# Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

		# results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

		# sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange','gray'])

		# ax = plt.gca()
		# handles, labels = ax.get_legend_handles_labels()
		# ax.legend(handles=handles[1:], labels=labels[1:])

		# print(gamma)
		# print(pcpca.sigma2_mle)
		# # plt.subplot(121)
		plot_df = X_reduced_df[X_reduced_df.condition.isin(["Pretransplant", "Posttransplant"])]
		sns.scatterplot(data=plot_df, x="PCPC1", y="PCPC2", hue="condition", alpha=0.1)
		plt.xlabel("PCPC3")
		plt.ylabel("PCPC4")
		# # plt.subplot(122)
		# # plot_df = X_reduced_df[X_reduced_df.condition.isin(["Posttransplant1"])]
		# # sns.scatterplot(data=plot_df, x="PCPC1", y="PCPC2", hue="condition", alpha=0.1)
		# plt.show()


	plt.savefig("../../../plots/scrnaseq/pcpca_singlecell_bmmc.png")
	
	plt.show()
	import ipdb; ipdb.set_trace()

	# import ipdb
	# ipdb.set_trace()
