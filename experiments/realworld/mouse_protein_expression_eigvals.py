import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../../models")
from cpca import CPCA


DATA_PATH = "../../data/mouse_protein_expression/clean/Data_Cortex_Nuclear.csv"
GAMMA = 0.5
N_COMPONENTS = 2

def compute_sample_covariance(data):
    """Compute sample covariance where data is a p x n matrix.
    """
    n = data.shape[1]
    cov = 1 / n * data @ data.T
    return cov


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

	# gamma_range = [0, 0.2, 0.5, 1.0, 10, 100, 1000]
	gamma_range = [10.0**x for x in range(5)]
	# plt.figure(figsize=(len(gamma_range) * 7, 5))

	neg_gamma_list = []
	for ii, gamma in enumerate(gamma_range):

		Cx, Cy = compute_sample_covariance(X), compute_sample_covariance(Y)
		C = Cx - gamma * Cy
		eigvals, U = np.linalg.eig(C)

		# plt.plot(-np.sort(-eigvals), label=gamma)
		neg_gamma_idx = np.where(-np.sort(-eigvals) < 0)[0][0] + 1
		neg_gamma_list.append(neg_gamma_idx)
		# print(-np.sort(-eigvals))


		# import ipdb; ipdb.set_trace()


	# plt.legend()
	# plt.xlabel("Index")
	# plt.ylabel("Eigenvalue")
	
	# plt.show()

	plt.figure(figsize=(7, 5))
	plt.scatter(gamma_range, neg_gamma_list)
	plt.xscale("log")
	plt.xlabel("Gamma")
	plt.ylabel("Index, first negative eigenvalue")
	plt.tight_layout()
	plt.savefig("../../plots/experiments/mouse_protein_expression/negative_eigenvalues.png")
	plt.show()

	# import ipdb
	# ipdb.set_trace()
