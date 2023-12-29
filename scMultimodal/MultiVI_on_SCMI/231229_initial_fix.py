# -*- coding: utf-8 -*-
"""
Created on 2023-12-29 (Fri) 15:09:23

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd

import scipy
import scipy.sparse
import scanpy as sc

import mudata
from mudata import AnnData, MuData

# %%
# inputs (atac)
atac_inputs = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_inputs_values_32606_HSC.sparse.npz")
#atac_inputs = atac_inputs.astype('float16', copy=False)
inputs_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_inputs_idxcol_32606_HSC.npz', allow_pickle=True)

# targets (rna)
rna_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)

# %% train test split for ATAC
train_size = 6510
samples = inputs_idx['index']
train_samples = samples[0:train_size] # 6510
test_samples = samples[train_size::] # 2790

atac_array = atac_inputs.toarray()
train_atac = atac_array[0:train_size,:] # (6510, 228942)
test_atac = atac_array[train_size:,:] # (2790, 228942)
test_atac = scipy.sparse.csr_matrix(test_atac) # update

# prep atac info
tmp_ids = list(inputs_idx['columns'])
atac_info = pd.DataFrame({'ID':tmp_ids,'modality':['Peaks']*len(tmp_ids)},index=tmp_ids)
atac_batch = pd.DataFrame({'batch_id':[1]*len(atac_info)},index=atac_info.index.tolist())

# %% train test split for RNA
rna_array = rna_targets.toarray() # (9300, 23418)
train_rna = rna_array[0:train_size,:]
test_rna = rna_array[train_size:,:]

# %% create paired data
"""
MultiVI requires the features to be ordered so that genes appear before genomic regions. This must be enforced by the user.
"""
paired_array = np.hstack([train_rna, train_atac]) # (6510, 23418+228942)
paired_columns = np.hstack([targets_idx['columns'],inputs_idx['columns']])

paired_csr = scipy.sparse.csr_matrix(paired_array)
paired_csr = paired_csr.astype('float16', copy=False)

# prep paired info
tmp_ids = list(targets_idx['columns'])+list(inputs_idx['columns'])
paired_info = pd.DataFrame({'ID':tmp_ids,'modality':['Gene Expression']*len(targets_idx['columns'])+['Peaks']*len(inputs_idx['columns'])},index=tmp_ids)
paired_batch = pd.DataFrame({'batch_id':[1]*len(paired_info)},index=paired_info.index.tolist())

# %% process for AnnData
adata_paired = AnnData(paired_csr)
adata_atac = AnnData(test_atac)

adata_paired.var = paired_info # add info
adata_paired.obs = paied_batch

adata_atac.var = atac_info # add info
adata_atac.obs = atac_batch
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

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

sc.set_figure_params(figsize=(4, 4), frameon=False)
torch.set_float32_matmul_precision("high")
save_dir = tempfile.TemporaryDirectory()

%config InlineBackend.print_figure_kwargs={"facecolor" : "w"}
%config InlineBackend.figure_format="retina"

# %%
# We can now use the organizing method from scvi to concatenate these anndata
adata_mvi = scvi.data.organize_multiome_anndatas(multi_anndata=adata_paired, atac_anndata=adata_atac)

display(adata_mvi.obs)

adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
display(adata_mvi.var)

# filter
print(adata_mvi.shape) # (9300, 252360)
sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))
print(adata_mvi.shape) # (9300, 111356)

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

# %%
tmp = sc.read_h5ad("/workspace/mnt/data1/MSCI/10x_multiome/paired.h5ad")

# %%
