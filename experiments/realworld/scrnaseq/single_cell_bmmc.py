from pcpca import PCPCA, CPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
from scipy.io import mmread
from sklearn.decomposition import PCA


DATA_DIR = "../../../data/singlecell_bmmc"
N_COMPONENTS = 10


if __name__ == "__main__":

    # Read in data
    # pretransplant1 = pd.read_csv(pjoin(DATA_DIR, "clean", "pretransplant1.csv"), index_col=0)
    # posttransplant1 = pd.read_csv(pjoin(DATA_DIR, "clean", "posttransplant1.csv"), index_col=0)

    pretransplant2 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "pretransplant2.csv"), index_col=0
    )
    posttransplant2 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "posttransplant2.csv"), index_col=0
    )

    healthy1 = pd.read_csv(pjoin(DATA_DIR, "clean", "healthy1.csv"), index_col=0)
    # healthy2 = pd.read_csv(pjoin(DATA_DIR, "clean", "healthy2.csv"), index_col=0)

    # Background is made up of healthy cells
    Y = healthy1.values  # pd.concat([healthy1, healthy2], axis=0).values

    X = pd.concat([pretransplant2, posttransplant2], axis=0).values
    X_labels = ["Pretransplant" for _ in range(pretransplant2.shape[0])]
    X_labels.extend(["Posttransplant" for _ in range(posttransplant2.shape[0])])
    X_labels = np.array(X_labels)
    assert X_labels.shape[0] == X.shape[0]

    # Standardize
    Y -= Y.mean(0)
    Y /= Y.std(0)
    Y = Y.T
    X -= X.mean(0)
    X /= X.std(0)
    X = X.T

    n, m = X.shape[1], Y.shape[1]

    X_df = pd.DataFrame(X.T)
    X_df["condition"] = X_labels

    import matplotlib

    font = {"size": 20}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    gamma_range = [0, 0.7, 0.9]

    plt.figure(figsize=(len(gamma_range) * 6, 5))

    for ii, gamma in enumerate(gamma_range):

        pcpca = PCPCA(gamma=n / m * gamma, n_components=N_COMPONENTS)
        X_reduced, Y_reduced = pcpca.fit_transform(X, Y)

        plt.subplot(1, len(gamma_range), ii + 1)
        if gamma == 0:
            plt.title(r"$\gamma^\prime$={} (PPCA)".format(gamma))
        else:
            plt.title(r"$\gamma^\prime$={}".format(gamma))

        X_reduced_df = pd.DataFrame(X_reduced.T[:, 2:4], columns=["PCPC1", "PCPC2"])
        X_reduced_df["condition"] = X_labels

        # Y_reduced_df = pd.DataFrame(Y_reduced.T[:, 2:4], columns=["PCPC1", "PCPC2"])
        # Y_reduced_df['condition'] = [
        #     "Background" for _ in range(Y_reduced_df.shape[0])]

        plot_df = X_reduced_df[
            X_reduced_df.condition.isin(["Pretransplant", "Posttransplant"])
        ]

        # plot_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)
        sns.scatterplot(
            data=plot_df,
            x="PCPC1",
            y="PCPC2",
            hue="condition",
            alpha=0.5,
            palette=["green", "orange"],
        )
        plt.xlabel("PCPC3")
        plt.ylabel("PCPC4")

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])

    plt.tight_layout()
    plt.savefig("../../../plots/scrnaseq/pcpca_singlecell_bmmc.png")

    plt.show()
    import ipdb

    ipdb.set_trace()
