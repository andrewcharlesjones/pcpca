import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../../models")
from pcpca import PCPCA
from cpca import CPCA


DATA_PATH = "../../data/mouse_protein_expression/clean/Data_Cortex_Nuclear.csv"
GAMMA = 0.5
N_COMPONENTS = 2
SUBSET_SIZE = 30


if __name__ == "__main__":

	# Read in data
	data = pd.read_csv(DATA_PATH)

	

	# Separate into background and foreground data
	# In this case,
	# background data is data from mice who did not receive shock therapty
	# foreground data is from mice who did receive shock therapy

	# Remove proteins with lots of NAs
	data = data.fillna(0)

	# Get names of proteins
	protein_names = data.columns.values[1:78]
	
	# Background
	Y_df = data[(data.Behavior == "C/S") & (data.Genotype == "Control") & (data.Treatment == "Saline")]

	# Subset to smaller dataset
	# chosen_idx = np.random.choice(np.arange(Y_df.shape[0]), replace=False, size=SUBSET_SIZE)
	# Y_df = Y_df.iloc[chosen_idx, :]
	Y = Y_df[protein_names].values
	Y -= Y.mean(0)
	Y /= Y.std(0)
	Y = Y.T


	# Foreground
	X_df = data[(data.Behavior == "S/C") & (data.Treatment == "Saline")]

	# Subset to smaller dataset
	# chosen_idx = np.random.choice(np.arange(X_df.shape[0]), replace=False, size=SUBSET_SIZE)
	# X_df = X_df.iloc[chosen_idx, :]
	X = X_df[protein_names].values
	X -= X.mean(0)
	X /= X.std(0)
	X = X.T

	n, m = X.shape[1], Y.shape[1]
	# import ipdb; ipdb.set_trace()

	import matplotlib
	font = {'size'   : 15}
	matplotlib.rc('font', **font)

	gamma = 0.1
	# gamma_range = [0, 0.1, 0.2, 0.3]
	# gamma_range = np.linspace(0, n/m-1e-2, 4)
	plt.figure(figsize=(len(subset_size_range) * 7, 17))

	subset_size_range = [20, 100, 200, X.shape[1]]
	for ii, subset_size in enumerate(subset_size_range):

		# gamma = n/m*0.5
		

		Y_df = data[(data.Behavior == "C/S") & (data.Genotype == "Control") & (data.Treatment == "Saline")]

		# Subset to smaller dataset
		# chosen_idx = np.random.choice(np.arange(Y_df.shape[0]), replace=False, size=subset_size)
		# Y_df = Y_df.iloc[chosen_idx, :]
		Y = Y_df[protein_names].values
		Y -= Y.mean(0)
		Y /= Y.std(0)
		Y = Y.T


		# Foreground
		X_df = data[(data.Behavior == "S/C") & (data.Treatment == "Saline")]

		# Subset to smaller dataset
		chosen_idx = np.random.choice(np.arange(X_df.shape[0]), replace=False, size=subset_size)
		X_df = X_df.iloc[chosen_idx, :]
		X = X_df[protein_names].values
		X -= X.mean(0)
		X /= X.std(0)
		X = X.T

		n, m = X.shape[1], Y.shape[1]

		pcpca = PCPCA(gamma=gamma, n_components=N_COMPONENTS)
		# pcpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
		X_reduced, Y_reduced = pcpca.fit_transform(X, Y)

		plt.subplot(2, len(gamma_range), ii+1)
		plt.title("n = {}\ngamma={}".format(subset_size, gamma))
		

		# Plot reduced foreground data
		X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
		X_reduced_df['Genotype'] = X_df.Genotype.values

		Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
		Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

		results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

		sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange','gray'])
		# plt.xlabel("CPC1")
		# plt.ylabel("CPC2")

		ax = plt.gca()
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles=handles[1:], labels=labels[1:])


		## Scale gamma
		pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
		# pcpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
		X_reduced, Y_reduced = pcpca.fit_transform(X, Y)

		plt.subplot(2, len(gamma_range), len(gamma_range)+ii+1)
		plt.title("m = {}\ngamma=n/m*{}".format(subset_size, gamma))
		

		# Plot reduced foreground data
		X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
		X_reduced_df['Genotype'] = X_df.Genotype.values

		Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
		Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

		results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

		sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange','gray'])
		# plt.xlabel("CPC1")
		# plt.ylabel("CPC2")

		ax = plt.gca()
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles=handles[1:], labels=labels[1:])


		


	plt.savefig("../../plots/experiments/mouse_protein_expression/pcpca_imbalanced_data_pcpca.png")
	
	plt.show()

	# import ipdb
	# ipdb.set_trace()
