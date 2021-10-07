from pcpca import CPCA
from pcpca import PCPCA, CPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


DATA_PATH = "../../../data/mouse_protein_expression/clean/Data_Cortex_Nuclear.csv"
N_COMPONENTS = 2


if __name__ == "__main__":

    # Read in data
    data = pd.read_csv(DATA_PATH)

    # Separate into background and foreground data
    # In this case,
    # background data is data from mice who did not receive shock therapty
    # foreground data is from mice who did receive shock therapy

    # Fill NAs
    data = data.fillna(0)

    # Get names of proteins
    # protein_names = data.columns.values[np.intersect1d(np.arange(1, 78), np.where(good_col_idx == True)[0])]
    protein_names = data.columns.values[1:78]

    # import ipdb
    # ipdb.set_trace()
    data.Genotype[data.Genotype == "Control"] = "Non-DS"
    data.Genotype[data.Genotype == "Ts65Dn"] = "DS"

    # Background
    Y_df = data[
        (data.Behavior == "C/S")
        & (data.Genotype == "Non-DS")
        & (data.Treatment == "Saline")
    ]
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

    # import ipdb; ipdb.set_trace()

    import matplotlib

    font = {"size": 20}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    gamma_range = [0, 0.5, 0.6]
    plt.figure(figsize=((len(gamma_range)) * 6, 5))

    # print(X[:5, :])
    # import ipdb; ipdb.set_trace()

    for ii, gamma in enumerate(gamma_range):

        pcpca = PCPCA(gamma=n / m * gamma, n_components=N_COMPONENTS)
        X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
        if pcpca.sigma2_mle < 0:
            raise ValueError("sigma^2 cannot be negative. PCPCA failed. ")
            import ipdb

            ipdb.set_trace()

        plt.subplot(1, len(gamma_range), ii + 1)
        if gamma == 0:
            plt.title(r"$\gamma^\prime$={} (PPCA)".format(gamma))
        else:
            plt.title(r"$\gamma^\prime$={}".format(gamma))

        # Plot reduced foreground data
        X_reduced_df = pd.DataFrame(X_reduced.T, columns=["PCPC1", "PCPC2"])
        X_reduced_df["Genotype"] = X_df.Genotype.values

        Y_reduced_df = pd.DataFrame(Y_reduced.T, columns=["PCPC1", "PCPC2"])
        Y_reduced_df["Genotype"] = ["Background" for _ in range(Y_reduced_df.shape[0])]

        if ii == len(gamma_range) - 1:

            # plt.subplot(1, len(gamma_range), ii + 1)

            # X_reduced_df.Genotype = "Foreground"
            results_df = pd.concat([X_reduced_df, Y_reduced_df], axis=0)

            sns.scatterplot(
                data=results_df,
                x="PCPC1",
                y="PCPC2",
                hue="Genotype",
                palette=["green", "orange", "gray"],
            )
        else:

            # pd.concat([X_reduced_df, Y_reduced_df], axis=0)
            results_df = X_reduced_df

            sns.scatterplot(
                data=results_df,
                x="PCPC1",
                y="PCPC2",
                hue="Genotype",
                palette=["green", "orange"],
            )
        plt.xlabel("PCPC1")
        plt.ylabel("PCPC2")

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])

        # plt.xlabel("PCPC1")
        # plt.ylabel("PCPC2")
        # plt.title(r"$\gamma^\prime$={}".format(gamma))

        # ax = plt.gca()
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=handles[1:], labels=labels[1:])

    plt.tight_layout()
    plt.savefig("../../../plots/mouse_protein_expression/mouse_pcpca_vary_gamma.png")

    plt.show()

    # import ipdb
    # ipdb.set_trace()
