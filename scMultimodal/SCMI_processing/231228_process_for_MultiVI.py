# -*- coding: utf-8 -*-
"""
Created on 2023-12-28 (Thu) 18:48:11

@author: I.Azuma
"""
# %%
import numpy as np
import scipy
import scipy.sparse

import mudata
from mudata import AnnData, MuData

# %%
# inputs (atac)
train_inputs = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_inputs_values_32606_HSC.sparse.npz")
train_inputs = train_inputs.astype('float16', copy=False)
inputs_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_inputs_idxcol_32606_HSC.npz', allow_pickle=True)

# targets (rna)
train_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)

# %%
atac = AnnData(train_inputs)
atac.obs_names = inputs_idx['index']
atac.var_names = inputs_idx['columns']

rna = AnnData(train_targets)
rna.obs_names = targets_idx['index']
rna.var_names = targets_idx['columns']

# %%
import gzip
import os
import tempfile
from pathlib import Path

import numpy as np
import pooch
import scanpy as sc
import scvi
import torch
# %%
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

sc.set_figure_params(figsize=(4, 4), frameon=False)
torch.set_float32_matmul_precision("high")
save_dir = tempfile.TemporaryDirectory()

%config InlineBackend.print_figure_kwargs={"facecolor" : "w"}
%config InlineBackend.figure_format="retina"
# %% Load data
"""
First we download a sample multiome dataset from 10X. We’ll use this throughout this tutorial. Importantly, MultiVI assumes that there are shared features between the datasets. This is trivial for gene expression datasets, which generally use the same set of genes as features. For ATAC-seq peaks, this is less trivial, and often requires preprocessing steps with other tools to get all datasets to use a shared set of peaks. That can be achieved with tools like SnapATAC, ArchR, and CellRanger in the case of 10X data.
"""
def download_data(save_path: str, fname: str = "pbmc_10k"):
    data_paths = pooch.retrieve(
        url="https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k_filtered_feature_bc_matrix.tar.gz",
        known_hash="872b0dba467d972aa498812a857677ca7cf69050d4f9762b2cd4753b2be694a1",
        fname=fname,
        path=save_path,
        processor=pooch.Untar(),
        progressbar=True,
    )
    data_paths.sort()

    for path in data_paths:
        with gzip.open(path, "rb") as f_in:
            with open(path.replace(".gz", ""), "wb") as f_out:
                f_out.write(f_in.read())

    return str(Path(data_paths[0]).parent)

data_path = download_data(save_dir.name)
# %% Data processing
# read multiomic data
adata = scvi.data.read_10x_multiome(data_path)
adata.var_names_make_unique()

n = 4004
adata_rna = adata[:n, adata.var.modality == "Gene Expression"].copy()
adata_paired = adata[n : 2 * n].copy()
adata_atac = adata[2 * n :, adata.var.modality == "Peaks"].copy()

# Filter size to avoid rendering error
narrow = 500
adata_rna = adata_rna[:narrow].copy()
adata_paired = adata_paired[:narrow].copy()
adata_atac = adata_atac[:narrow].copy()
# %% save and load
# save
# adata_paired.write_h5ad("/workspace/mnt/data1/MSCI/10x_multiome/paired.h5ad",compression='gzip')

# load
tmp = sc.read_h5ad("/workspace/mnt/data1/MSCI/10x_multiome/paired.h5ad")

# %% common features in default setting
atac_common = set(atac.var_names) & set(adata_atac.var['ID'].tolist()) # 34

atac1 = set([t.split(':')[0] for t in adata_atac.var['ID'].tolist()])
atac2 = set([t.split(':')[0] for t in atac.var_names.tolist()])
atac_common = set(atac1) & set(atac2)
