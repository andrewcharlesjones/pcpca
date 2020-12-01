import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import seaborn as sns


def _compute_sample_covariance(data):
    """Compute sample covariance where data is a p x n matrix.
    """
    n = data.shape[1]
    cov = 1 / n * data @ data.T
    return cov


n = 100
p = 30
k = 5

W_true = np.random.normal()


