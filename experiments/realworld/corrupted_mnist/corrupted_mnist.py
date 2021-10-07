from pcpca import PCPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
from scipy.stats import multivariate_normal
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import silhouette_score
import sys
sys.path.append("../../../clvm")
from clvm import CLVM


N_COMPONENTS = 2
n_to_plot = 300

if __name__ == "__main__":

    # Load data
    data_dir = "../../../data/corrupted_mnist"
    X = np.load(pjoin(data_dir, "foreground.npy"))
    Y = np.load(pjoin(data_dir, "background.npy"))
    X_labels = np.load(pjoin(data_dir, "foreground_labels.npy"))
    digits_test = np.load(pjoin(data_dir, "mnist_digits_test.npy"))

    X_mean, Y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
    X = (X - X_mean) / np.std(X, axis=0)
    Y = (Y - Y_mean) / np.std(Y, axis=0)

    digits_test_mean = np.mean(digits_test, axis=0)
    digits_test = (digits_test - digits_test_mean) / np.std(digits_test, axis=0)

    X, Y = X.T, Y.T
    digits_test = digits_test.T
    n = X.shape[1]
    m = Y.shape[1]

    import matplotlib

    font = {"size": 20}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    n_methods = 3
    plt.figure(figsize=(5 * n_methods, 10))

    # PPCA
    GAMMA = 0

    # Fit model
    pcpca = PCPCA(gamma=GAMMA, n_components=N_COMPONENTS)
    X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
    print("sigma2: {}".format(pcpca.sigma2_mle))

    X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PPC1", "PPC2"])
    X_reduced_df["Digit"] = X_labels

    ss_ppca = silhouette_score(
        X=X_reduced_df[["PPC1", "PPC2"]] / X_reduced_df[["PPC1", "PPC2"]].std(0),
        labels=X_labels,
    )
    print("SS PPCA: {}".format(ss_ppca))

    plt.subplot(231)
    sns.scatterplot(
        data=X_reduced_df,
        x="PPC1",
        y="PPC2",
        hue="Digit",
        alpha=0.8,
        palette=["green", "orange"],
    )
    plt.title("PPCA")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])

    # Simulate new data
    plt.subplot(234)
    ax = plt.gca()
    ii = 0
    for _ in range(n_to_plot):
        z = multivariate_normal(np.zeros(N_COMPONENTS), np.eye(N_COMPONENTS)).rvs()
        z = np.expand_dims(z, 1)
        xhat = (pcpca.W_mle @ z).T + X_mean

        ax.scatter(z[0, 0], z[1, 0])
        curr_im = np.reshape(xhat.squeeze(), [28, 28])
        im_object = OffsetImage(curr_im, cmap="gray", interpolation="bicubic")
        ab = AnnotationBbox(im_object, (z[0, 0], z[1, 0]), frameon=False)
        ax.add_artist(ab)

    plt.xlabel(r"$z_1$")
    plt.ylabel(r"$z_2$")
    plt.title("Generated images (PPCA)")

    #############################
    ########### CLVM ############
    #############################
    ## Fit CLVM
    clvm = CLVM(
        data_dim=X.shape[0],
        n_bg=m,
        n_fg=n,
        latent_dim_shared=N_COMPONENTS,
        latent_dim_fg=N_COMPONENTS,
    )
    clvm.init_model()
    clvm.fit_model(Y, X, n_iters=100000)
    tx = clvm.qtx_mean.numpy().T

    X_reduced_df = pd.DataFrame(tx, columns=["CLV1", "CLV2"])
    X_reduced_df["Digit"] = X_labels

    # ss_pcpca = silhouette_score(
    #     X=X_reduced_df[["PCPC1", "PCPC2"]] / X_reduced_df[["PCPC1", "PCPC2"]].std(0),
    #     labels=X_labels,
    # )
    # print("SS PCPCA: {}".format(ss_pcpca))

    plt.subplot(232)
    sns.scatterplot(
        data=X_reduced_df,
        x="CLV1",
        y="CLV2",
        hue="Digit",
        alpha=0.8,
        palette=["green", "orange"],
    )
    plt.title(r"CLVM")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])

    ## Plot generated images

    plt.subplot(235)
    ax = plt.gca()

    W_fitted = clvm.W.numpy().T
    tx_sampled = np.random.normal(loc=0, scale=1, size=(n_to_plot, N_COMPONENTS))
    clvm_samples = tx_sampled @ W_fitted
    # import ipdb; ipdb.set_trace()

    for jj in range(n_to_plot):
        curr_im = np.reshape(clvm_samples[jj], [28, 28])

        ax.scatter(tx_sampled[jj, 0], tx_sampled[jj, 1])
        im_object = OffsetImage(curr_im, cmap="gray", interpolation="bicubic")
        ab = AnnotationBbox(im_object, (tx_sampled[jj, 0], tx_sampled[jj, 1]), frameon=False)
        ax.add_artist(ab)

    plt.xlabel(r"$t_1$")
    plt.ylabel(r"$t_2$")
    plt.title(r"Generated images (CLVM)".format(GAMMA))


    
    # np.reshape(clvm_samples[0], [28, 28])
    # plt.imshow(np.reshape(clvm_samples[0], [28, 28]), cmap="gray")
    # plt.show()
    # import ipdb; ipdb.set_trace()

    #############################
    ########### PCPCA ###########
    #############################
    GAMMA = 0.8

    # Fit model
    pcpca = PCPCA(gamma=GAMMA, n_components=N_COMPONENTS)
    X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
    # print("sigma2: {}".format(pcpca.sigma2_mle))

    X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
    X_reduced_df["Digit"] = X_labels

    ss_pcpca = silhouette_score(
        X=X_reduced_df[["PCPC1", "PCPC2"]] / X_reduced_df[["PCPC1", "PCPC2"]].std(0),
        labels=X_labels,
    )
    print("SS PCPCA: {}".format(ss_pcpca))

    plt.subplot(233)
    sns.scatterplot(
        data=X_reduced_df,
        x="PCPC1",
        y="PCPC2",
        hue="Digit",
        alpha=0.8,
        palette=["green", "orange"],
    )
    plt.title(r"PCPCA, $\gamma^\prime$={}".format(GAMMA))
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])

    # Simulate new data
    
    plt.subplot(236)
    ax = plt.gca()
    ii = 0
    for _ in range(n_to_plot):
        z = multivariate_normal(np.zeros(N_COMPONENTS), np.eye(N_COMPONENTS)).rvs()
        z = np.expand_dims(z, 1)
        xhat = (pcpca.W_mle @ z).T + X_mean

        ax.scatter(z[0, 0], z[1, 0])
        curr_im = np.reshape(xhat.squeeze(), [28, 28])
        im_object = OffsetImage(curr_im, cmap="gray", interpolation="bicubic")
        ab = AnnotationBbox(im_object, (z[0, 0], z[1, 0]), frameon=False)
        ax.add_artist(ab)

    plt.xlabel(r"$z_1$")
    plt.ylabel(r"$z_2$")
    plt.title(r"Generated images (PCPCA)") #, $\gamma^\prime$={})".format(GAMMA))

    plt.tight_layout()
    plt.savefig("../../../plots/corrupted_mnist/ppca_pcpca_comparison_mnist.png")
    plt.show()
    plt.close()

    

    

    import ipdb

    ipdb.set_trace()
