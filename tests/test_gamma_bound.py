from pcpca import PCPCA
import numpy as np
from os.path import join as pjoin
import pandas as pd
from scipy.stats import multivariate_normal as mvn

def test_gamma_bound_toy():
	data_dir = "./data/toy"
	X = pd.read_csv(pjoin(data_dir, "foreground.csv"), header=None)
	Y = pd.read_csv(pjoin(data_dir, "background.csv"), header=None)
	pcpca = PCPCA(n_components=1)
	gamma_bound = pcpca.get_gamma_bound(X, Y)
	assert gamma_bound > 0
	assert ~isinstance(gamma_bound, complex)

def test_gamma_bound_random():
	n_repeats = 10
	for _ in range(n_repeats):
		n = 100
		p = 10
		X = np.random.normal(size=(n, p))
		Y = np.random.normal(size=(n, p))
		pcpca = PCPCA(n_components=4)
		gamma_bound = pcpca.get_gamma_bound(X, Y)
		assert gamma_bound > 0
		assert ~isinstance(gamma_bound, complex)

if __name__ == "__main__":
	test_gamma_bound_random()
	