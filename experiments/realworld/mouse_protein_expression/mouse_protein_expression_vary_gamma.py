import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

	

	# Separate into background and foreground data
	# In this case,
	# background data is data from mice who did not receive shock therapty
	# foreground data is from mice who did receive shock therapy

	# First, remove mice that got drug treatment too
	# data = data[data.Treatment != "Memantine"]

	# Remove proteins with lots of NAs
	# good_col_idx = data.isna().sum(0).values < 10
	data = data.fillna(0)

	# Get names of proteins
	# protein_names = data.columns.values[np.intersect1d(np.arange(1, 78), np.where(good_col_idx == True)[0])]
	protein_names = data.columns.values[1:78]


	# data = data.iloc[:, good_col_idx]

	
	# Fill in nans with column means for now
	# data = data.fillna(data.mean())

	# Background
	Y_df = data[(data.Behavior == "C/S") & (data.Genotype == "Control") & (data.Treatment == "Saline")]
	# Y_df = Y_df.iloc[:20, :]
	Y = Y_df[protein_names].values
	Y -= Y.mean(0)
	Y /= Y.std(0)
	Y = Y.T


	# Foreground
	X_df = data[(data.Behavior == "S/C") & (data.Treatment == "Saline")]
	X = X_df[protein_names].values
	X -= X.mean(0)
	X /= X.std(0)
	X = X.T

	n, m = X.shape[1], Y.shape[1]

	

	import matplotlib
	font = {'size'   : 15}
	matplotlib.rc('font', **font)

	# gamma_range = [0, 5, 10, 20, 40, 200, 300]
	gamma_range = [0, 0.5, 0.6, 0.7, 0.9]
	plt.figure(figsize=(len(gamma_range) * 7, 5))

	for ii, gamma in enumerate(gamma_range):
		# gamma = gamma
		pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
		X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
		# cpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
		# X_reduced, Y_reduced = cpca.fit_transform(X, Y)

		plt.subplot(1, len(gamma_range), ii+1)
		if gamma == 0:
			plt.title("gamma={}*n/m (PPCA)\nsigma2={}".format(gamma, round(pcpca.sigma2_mle, 2)))
		else:
			plt.title("gamma={}*n/m\nsigma2={}".format(gamma, round(pcpca.sigma2_mle, 2)))

		# Plot reduced foreground data
		X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
		X_reduced_df['Genotype'] = X_df.Genotype.values

		Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
		Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

		results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

		sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange','gray'])
		plt.xlabel("CPC1")
		plt.ylabel("CPC2")

		ax = plt.gca()
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles=handles[1:], labels=labels[1:])


	plt.savefig("../../../plots/mouse_protein_expression/mouse_pcpca_vary_gamma.png")
	
	plt.show()

	# import ipdb
	# ipdb.set_trace()
