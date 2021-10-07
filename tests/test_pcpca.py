from pcpca import PCPCA
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import numpy as np

def test_ppca():
	X = multivariate_normal.rvs(np.zeros(2), np.array([[2.7, 2.6], [2.6, 2.7]]), size=200)
	Y = multivariate_normal.rvs(np.zeros(2), np.array([[2.7, 2.6], [2.6, 2.7]]), size=200)

	## PPCA via PCPCA + gamma=0
	pcpca = PCPCA(n_components=1, gamma=0)
	pcpca.fit(X.T, Y.T)
	pcpca_W = np.squeeze(pcpca.W_mle)
	pcpca_W_normalized = pcpca_W / np.linalg.norm(pcpca_W, ord=2)

	## PCA
	pca = PCA(n_components=1).fit(np.concatenate([X, Y], axis=0))
	pca_W = np.squeeze(pca.components_)
	pca_W_normalized = pca_W / np.linalg.norm(pca_W, ord=2)

	## Check if PPCA and PCA agree (account for negative)
	assert np.allclose(pcpca_W_normalized, pca_W_normalized, rtol=0.1) or np.allclose(-pcpca_W_normalized, pca_W_normalized, rtol=0.1)

if __name__ == "__main__":
	test_ppca()