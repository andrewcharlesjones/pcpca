from pcpca import CPCA
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import numpy as np

def test_cpca_failure():
	X = multivariate_normal.rvs(np.zeros(2), np.array([[2.7, 2.6], [2.6, 2.7]]), size=200)
	Y = multivariate_normal.rvs(np.zeros(2), np.array([[2.7, 2.6], [2.6, 2.7]]), size=200)

	cpca = CPCA(n_components=2, gamma=10000)
	try:
		cpca.fit(X.T, Y.T)
	except:
		assert True
		return

	assert False


if __name__ == "__main__":
	test_cpca_failure()