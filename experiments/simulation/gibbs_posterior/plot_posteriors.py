import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin

RESULTS_DIR = "./out"
N_W_SAMPLES = 500


mu1 = np.array([-2, 2])
mu2 = np.array([2, -2])
cov = [
    [2.7, 2.6],
    [2.6, 2.7]
]
cov_mixture = cov + 0.5 * (np.outer(mu1, mu1) + np.outer(mu2, mu2))

import ipdb; ipdb.set_trace()

fnames = os.listdir(RESULTS_DIR)
ns = np.unique([int(x.split("_")[-1].split(".")[0]) for x in fnames])


print(ns)
plt.figure(figsize=(7*len(ns), 5))
for ii, n in enumerate(ns):

	# load data
	data_file = pjoin(RESULTS_DIR, "data_{}.csv".format(n))
	w_file = pjoin(RESULTS_DIR, "W_samples_{}.csv".format(n))

	data = pd.read_csv(data_file, index_col=0)
	data_fg = data[data.condition == "Foreground"][["x1", "x2"]].values
	data_bg = data[data.condition == "Background"][["x1", "x2"]].values
	Ws = pd.read_csv(w_file, index_col=0)
	

	# plot data
	plt.subplot(1, len(ns), ii+1)
	plt.xlim([-7, 7])
	plt.ylim([-7, 7])
	plt.title("n=m={}".format(n))
	plt.scatter(data_fg[:n//2, 0], data_fg[:n//2, 1], alpha=0.5, label="Foreground group 1", s=50, color="green")
	plt.scatter(data_fg[n//2:, 0], data_fg[n//2:, 1], alpha=0.5, label="Foreground group 2", s=50, color="orange")
	plt.scatter(data_bg[:, 0], data_bg[:, 1], alpha=0.5, label="Background", s=50, color="gray")
	plt.legend(prop={'size': 10})

	# Plot W samples
	axes = plt.gca()

	rand_idx = np.random.choice(Ws.shape[0], N_W_SAMPLES)
	x_vals = np.array(axes.get_xlim())
	for idx in rand_idx:
		slope = Ws.values[idx, 1] / Ws.values[idx, 0]
		y_vals = slope * x_vals
		plt.plot(x_vals, y_vals, '--', c="blue", alpha=0.1)

plt.savefig("../../../plots/simulated/gibbs_samples_varyn.png")
plt.show()
import ipdb; ipdb.set_trace()