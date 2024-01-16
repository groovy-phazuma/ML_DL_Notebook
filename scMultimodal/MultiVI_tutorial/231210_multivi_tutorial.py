# -*- coding: utf-8 -*-
"""
Created on 2023-12-10 (Sun) 21:01:12

MultiVI tutorial

References
- https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/MultiVI_tutorial.html


! pip install pooch
! pip install scvi-tools
! pip install scanpy

@author: I.Azuma
"""
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

"""
We can now split the dataset to three datasets, and remove a modality from two of them, so we'll have equal-sized (4004 cells in each) datasets with multiome, expression-only, and accessibility-only observations:
"""

# split to three datasets by modality (RNA, ATAC, Multiome), and corrupt data
# by remove some data to create single-modality data
n = 4004
adata_rna = adata[:n, adata.var.modality == "Gene Expression"].copy()
adata_paired = adata[n : 2 * n].copy()
adata_atac = adata[2 * n :, adata.var.modality == "Peaks"].copy()

# Filter size to avoid rendering error
narrow = 500
adata_rna = adata_rna[:narrow].copy()
adata_paired = adata_paired[:narrow].copy()
adata_atac = adata_atac[:narrow].copy()

# We can now use the organizing method from scvi to concatenate these anndata
adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna, adata_atac)

display(adata_mvi.obs)
"""
MultiVI requires the features to be ordered so that genes appear before genomic regions. This must be enforced by the user.
"""
adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
display(adata_mvi.var)

# filter
print(adata_mvi.shape)
sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))
print(adata_mvi.shape)


# %% Setup and Training MultiVI
scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")

model = scvi.model.MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
)
model.view_anndata_setup()
model.train()
# %% Load model
model_dir = os.path.join(save_dir.name, "multivi_pbmc10k")
model.save(model_dir, overwrite=True)
model = scvi.model.MULTIVI.load(model_dir, adata=adata_mvi)

#%% Visualizing the latent space
MULTIVI_LATENT_KEY = "X_multivi"

adata_mvi.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation()
sc.pp.neighbors(adata_mvi, use_rep=MULTIVI_LATENT_KEY)
sc.tl.umap(adata_mvi, min_dist=0.2)
sc.pl.umap(adata_mvi, color="modality")

# %% Impute missing modality
imputed_expression = model.get_normalized_expression()

gene_idx = np.where(adata_mvi.var.index == "CD3G")[0]
adata_mvi.obs["CD3G_imputed"] = imputed_expression.iloc[:, gene_idx]
sc.pl.umap(adata_mvi, color="CD3G_imputed")
# %% memo
# ペア、RNAのみ、ATACのみのAnnDataを統合
adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna, adata_atac)
# モデルにセット。バッチ補正。
scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
model = scvi.model.MULTIVI(adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
)
# 学習
model.train()
# 潜在表現の獲得と欠損モダリティーの補完
latent_key = model.get_latent_representation()
imputed_expression = model.get_normalized_expression()