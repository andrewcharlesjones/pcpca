import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
from scipy.io import mmread

DATA_DIR = "./singlecell_bmmc"
N_GENES = 500


def get_data_df(subdata_dir):
    data = mmread(pjoin(DATA_DIR, subdata_dir, "hg19/matrix.mtx")).toarray()
    data = np.log(data + 1)
    genes = pd.read_table(pjoin(DATA_DIR, subdata_dir, "hg19/genes.tsv"), header=None)
    barcodes = pd.read_table(
        pjoin(DATA_DIR, subdata_dir, "hg19/barcodes.tsv"), header=None
    )
    data_df = pd.DataFrame(
        data, index=genes.iloc[:, 0].values, columns=barcodes.iloc[:, 0].values
    )

    data_df = data_df.iloc[:, np.sum(data_df.values, axis=0) != 0]
    data_df = data_df.iloc[np.sum(data_df.values, axis=1) != 0, :]
    return data_df.transpose()


## Load all data
pretransplant1 = get_data_df("AML027_pretransplant_BMMCs")
posttransplant1 = get_data_df("AML027_posttransplant_BMMCs")
pretransplant2 = get_data_df("AML035_pretransplant_BMMCs")
posttransplant2 = get_data_df("AML035_posttransplant_BMMCs")

healthy1 = get_data_df("Frozen_BMMCs_HealthyControl1")
healthy2 = get_data_df("Frozen_BMMCs_HealthyControl2")

## Subset to shared genes
shared_genes = pretransplant1.columns.values
for curr_df in [posttransplant1, pretransplant2, posttransplant2, healthy1, healthy2]:
    shared_genes = np.intersect1d(shared_genes, curr_df.columns.values)

## Combine into one dataframe
stacked_df = pretransplant1[shared_genes]
for curr_df in [
    posttransplant1,
    pretransplant2,
    posttransplant2,
]:  # , healthy1, healthy2]:
    stacked_df = pd.concat([stacked_df, curr_df[shared_genes]], axis=0)
print("Total of {} cells and {} genes".format(stacked_df.shape[0], stacked_df.shape[1]))

## Subset to most variable genes
gene_means = np.mean(stacked_df.values, axis=0)
gene_vars = np.var(stacked_df.values, axis=0)
gene_dispersions = gene_vars / gene_means
top_idx = np.argsort(-gene_dispersions)[:N_GENES]
top_genes = stacked_df.columns.values[top_idx]

print("Saving {} genes".format(top_genes.shape[0]))


## Save
pretransplant1[top_genes].to_csv(pjoin(DATA_DIR, "clean", "pretransplant1.csv"))
posttransplant1[top_genes].to_csv(pjoin(DATA_DIR, "clean", "posttransplant1.csv"))

pretransplant2[top_genes].to_csv(pjoin(DATA_DIR, "clean", "pretransplant2.csv"))
posttransplant2[top_genes].to_csv(pjoin(DATA_DIR, "clean", "posttransplant2.csv"))

healthy1[top_genes].to_csv(pjoin(DATA_DIR, "clean", "healthy1.csv"))
healthy2[top_genes].to_csv(pjoin(DATA_DIR, "clean", "healthy2.csv"))
# import ipdb; ipdb.set_trace()
