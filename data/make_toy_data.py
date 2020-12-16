import numpy as np
from scipy.stats import multivariate_normal
from os.path import join as pjoin
import pandas as pd

# Toy data with two foreground subgroups
def generate_toy_data(n, m, cov):
	
	# Background
	Y = multivariate_normal.rvs([0, 0], cov, size=m)

	# Foreground groups a and b
	Xa = multivariate_normal.rvs([-1, 1], cov, size=n//2)
	Xb = multivariate_normal.rvs([1, -1], cov, size=n//2)
	X = np.concatenate([Xa, Xb], axis=0)

	# Make them p by n and p by m
	X, Y = X.T, Y.T
	return X, Y

if __name__ == "__main__":
	n, m = 200, 200
	cov = [
	    [2.7, 2.6],
	    [2.6, 2.7]
	]
	X, Y = generate_toy_data(n, m, cov)
	pd.DataFrame(X).to_csv(pjoin("toy", "foreground.csv"), index=False, header=False)
	pd.DataFrame(Y).to_csv(pjoin("toy", "background.csv"), index=False, header=False)
