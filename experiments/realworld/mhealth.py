import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
import seaborn as sns
import sys
sys.path.append("../../models")
from pcpca import PCPCA
from cpca import CPCA

N_COMPONENTS = 2


mhealth_dfs = []
for ii in range(1, 11):
    a = pd.read_table("../../data/mhealth/mHealth_subject{}.log".format(ii), header=None)
    mhealth_dfs.append(a)


data = pd.concat(mhealth_dfs, axis=0)

Y = data[data.iloc[:, -1] == 3]
X = data[data.iloc[:, -1].isin([8, 9])]
X_labels = X.iloc[:, -1].values

X = X.iloc[:, :-1]
Y = Y.iloc[:, :-1]

X -= X.mean(0)
X /= X.std(0)
X = X.T

Y -= Y.mean(0)
Y /= Y.std(0)
Y = Y.T

X_idx = np.random.choice(np.arange(X.shape[1]), replace=False, size=200)
X = X.values[:, X_idx]
X_labels = X_labels[X_idx]
Y = Y.values[:, np.random.choice(np.arange(Y.shape[1]), replace=False, size=200)]

n, m = X.shape[1], Y.shape[1]

X_df = pd.DataFrame(X.T)
X_df['condition'] = ["squatting" if x == 8 else "cycling" for x in X_labels]

import ipdb; ipdb.set_trace()

import matplotlib
font = {'size'   : 15}
matplotlib.rc('font', **font)

# gamma_range_cpca = [0, 5, 10, 20, 40]
# gamma_range_cpca.extend(np.arange(101, 400, 5))
# gamma_range_cpca.extend(np.arange(401, 420))
gamma_range_cpca = np.linspace(0, 2000, 50)
# import ipdb; ipdb.set_trace()
# gamma_range = [10, 400, 420]
gamma_range_pcpca = list(np.linspace(0, 0.99, 50)) # [0, 0.5, 0.6, 0.7, 0.9]
gamma_range_pcpca.extend(np.arange(0.9, 1.3, 0.03))

rand_scores_cpca = []
cpca_gamma_plot_list = []
for ii, gamma in enumerate(gamma_range_cpca):
	# gamma = gamma
	# pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
	# X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
	cpca = CPCA(gamma=gamma, n_components=N_COMPONENTS)
	X_reduced, Y_reduced = cpca.fit_transform(X, Y)

	X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
	
	try:
		kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
	except:
		cpca_fail_gamma = gamma
		break
	cpca_gamma_plot_list.append(gamma)

	true_labels = pd.factorize(X_df.condition)[0]
	# estimated_labels1 = kmeans.labels_
	# estimated_labels2 = ~kmeans.labels_
	# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
	# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
	# rand_score = max(rand_score1, rand_score2)
	rand_score = silhouette_score(X=X_reduced.T, labels=true_labels)
	print("gamma={}, rand score={}".format(gamma, rand_score))
	rand_scores_cpca.append(rand_score)

rand_scores_pcpca = []
pcpca_gamma_plot_list = []
for ii, gamma in enumerate(gamma_range_pcpca):
	gamma = gamma
	pcpca = PCPCA(gamma=n/m*gamma, n_components=N_COMPONENTS)
	X_reduced, Y_reduced = pcpca.fit_transform(X, Y)

	if pcpca.sigma2_mle <= 0:
		pcpca_fail_gamma = gamma
		break

	X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
	kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)
	pcpca_gamma_plot_list.append(gamma)

	true_labels = pd.factorize(X_df.condition)[0]
	# estimated_labels1 = kmeans.labels_
	# estimated_labels2 = ~kmeans.labels_
	# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
	# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
	# rand_score = max(rand_score1, rand_score2)
	rand_score = silhouette_score(X=X_reduced.T, labels=true_labels)
	print("gamma=n/m*{}, rand score={}".format(gamma, rand_score))
	rand_scores_pcpca.append(rand_score)


	# pcpca = PCPCA(gamma=gamma, n_components=N_COMPONENTS)
	# X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
	
	# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T
	# kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced.T)

	# true_labels = pd.factorize(X_df.Genotype)[0]
	# estimated_labels1 = kmeans.labels_
	# estimated_labels2 = ~kmeans.labels_
	# rand_score1 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels1)
	# rand_score2 = adjusted_rand_score(labels_true=true_labels, labels_pred=estimated_labels2)
	# rand_score = max(rand_score1, rand_score2)
	# print("gamma={}, rand score={}".format(gamma, rand_score))
	# rand_scores_cpca.append(rand_score)
	# import ipdb; ipdb.set_trace()


# 	plt.subplot(1, len(gamma_range), ii+1)
	# if gamma == 0:
	# 	plt.title("gamma={}*n/m (PPCA)\nsigma2={}".format(gamma, round(pcpca.sigma2_mle, 2)))
	# else:
	# 	plt.title("gamma={}*n/m\nsigma2={}".format(gamma, round(pcpca.sigma2_mle, 2)))

	# Plot reduced foreground data
	# X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
	# X_reduced_df['Genotype'] = X_df.Genotype.values # [str(x) for x in kmeans.labels_]

	# Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
	# Y_reduced_df['Genotype'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

	# # results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

	# sns.scatterplot(data=X_reduced_df, x="PCPC1", y="PCPC2", hue="Genotype", palette=['green','orange'])
	# plt.xlabel("CPC1")
	# plt.ylabel("CPC2")
	# plt.show()


plt.figure(figsize=(14, 14))
plt.subplot(221)
plt.plot(cpca_gamma_plot_list, rand_scores_cpca, '-o', linewidth=2)
plt.title("CPCA")
plt.ylim([0, 1])
plt.axvline(cpca_fail_gamma, color="black", linestyle="--")
plt.xlabel("gamma")
plt.ylabel("Adjusted Rand index")
plt.subplot(222)
plt.plot(pcpca_gamma_plot_list, rand_scores_pcpca, '-o', linewidth=2)
plt.title("PCPCA")
plt.ylim([0, 1])
plt.axvline(pcpca_fail_gamma, color="black", linestyle="--")
plt.xlabel("gamma * m/n")
plt.ylabel("Adjusted Rand index")
# plt.tight_layout()




plt.subplot(223)
cpca = CPCA(gamma=cpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
X_reduced, Y_reduced = cpca.fit_transform(X, Y)
# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

plt.title("CPCA, gamma={}".format(round(cpca_gamma_plot_list[-1], 2)))
X_reduced_df = pd.DataFrame(X_reduced.T)
X_reduced_df.columns = ["PCPC1", "PCPC2"]
X_reduced_df['condition'] = ["squatting" if x == 8 else "cycling" for x in X_labels]

Y_reduced_df = pd.DataFrame(Y_reduced.T)
Y_reduced_df.columns = ["PCPC1", "PCPC2"]
Y_reduced_df['condition'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue=results_df.condition.values, palette=['green','orange','gray'])

plt.xlabel("CPC1")
plt.ylabel("CPC2")


plt.subplot(224)
pcpca = PCPCA(gamma=n/m*pcpca_gamma_plot_list[-1], n_components=N_COMPONENTS)
X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
# X_reduced = (X_reduced.T / X_reduced.T.std(0)).T

plt.title("PCPCA, gamma*m/n={}".format(round(pcpca_gamma_plot_list[-1], 2)))
X_reduced_df = pd.DataFrame(X_reduced.T)
X_reduced_df.columns = ["PCPC1", "PCPC2"]
X_reduced_df['condition'] = ["squatting" if x == 8 else "cycling" for x in X_labels]

Y_reduced_df = pd.DataFrame(Y_reduced.T)
Y_reduced_df.columns = ["PCPC1", "PCPC2"]
Y_reduced_df['condition'] = ["Background" for _ in range(Y_reduced_df.shape[0])]

results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

sns.scatterplot(data=results_df, x="PCPC1", y="PCPC2", hue=results_df.condition.values, palette=['green','orange','gray'])


plt.savefig("../../plots/experiments/rand_index_comparison_mhealth.png")
plt.show()

import ipdb; ipdb.set_trace()