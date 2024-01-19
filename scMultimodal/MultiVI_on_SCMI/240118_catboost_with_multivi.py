# -*- coding: utf-8 -*-
"""
Created on 2024-01-19 (Fri) 00:12:56

n_latent = 128

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

atac_batch = pd.DataFrame(index=test_samples)
atac_batch['batch_id']=[1]*len(test_samples)

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

# prep paired info
tmp_ids = list(targets_idx['columns'])+list(inputs_idx['columns'])
paired_info = pd.DataFrame({'ID':tmp_ids,'modality':['Gene Expression']*len(targets_idx['columns'])+['Peaks']*len(inputs_idx['columns'])},index=tmp_ids)

paired_batch = pd.DataFrame(index=train_samples)
paired_batch['batch_id'] = [1]*len(train_samples)

# %% process for AnnData
paired_csr = paired_csr.astype('int64',copy=False)
test_atac = test_atac.astype('int64',copy=False)

adata_paired = AnnData(paired_csr)
adata_atac = AnnData(test_atac)

adata_paired.var = paired_info # add info
adata_paired.obs = paired_batch

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

obs_info = adata_mvi.obs
display(adata_mvi.obs)

adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
display(adata_mvi.var)

# filter
print(adata_mvi.shape) # (9300, 252360)
sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))
print(adata_mvi.shape) # (9300, 111356)

"""
AnnData object with n_obs × n_vars = 9300 × 111356
    obs: 'batch_id', 'modality', '_indices', '_scvi_batch', '_scvi_labels'
    var: 'ID', 'modality', 'n_cells'
    uns: '_scvi_uuid', '_scvi_manager_uuid'
"""

# %% Setup and Training MultiVI
scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")

model = scvi.model.MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
    n_latent=128
)
model.view_anndata_setup()
model.train()

# %% Load model
model_dir = os.path.join(save_dir.name, "multivi_pbmc10k")
model.save(model_dir, overwrite=True)
model = scvi.model.MULTIVI.load(model_dir, adata=adata_mvi)

# %% visualize the latent space
MULTIVI_LATENT_KEY = "X_multivi"

adata_mvi.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation()
sc.pp.neighbors(adata_mvi, use_rep=MULTIVI_LATENT_KEY)
sc.tl.umap(adata_mvi, min_dist=0.2)
sc.pl.umap(adata_mvi, color="modality")

latent_rep = model.get_latent_representation()
latent_rep = pd.DataFrame(latent_rep)

# .to_csv('/workspace/mnt/data1/MSCI/results/240119/latent_rep_9300x128.csv')

# %% catboost
import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

import scipy
import scipy.sparse

import gc
import pickle
import warnings
warnings.filterwarnings('ignore')

def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    if y_true.shape != y_pred.shape:
        print(y_true.shape)
        print(y_pred_shape)
        raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)

# targets (rna)
rna_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)

# %% Inputs (latent representation obtained from MultiVI)
inputs = pd.read_csv('/workspace/mnt/data1/MSCI/results/240119/latent_rep_9300x128.csv',index_col=0)
train_inputs = np.array(inputs)

# %% RNA compression
# targets (rna)
rna_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)

print('start targets SVD')
pca2 = TruncatedSVD(n_components=128, random_state=42)
train_target = pca2.fit_transform(rna_targets)
print(pca2.explained_variance_ratio_.sum())

# %% split into train and test data
train_size = 6510
train_X = train_inputs[0:train_size,:]
test_X = train_inputs[train_size:,:]
train_y = rna_targets[0:train_size,:]
test_y = rna_targets[train_size:,:]

# %%
from catboost import CatBoostRegressor
params = {'learning_rate': 0.1, 
          'depth': 7, 
          'l2_leaf_reg': 4, 
          'loss_function': 'MultiRMSE', 
          'eval_metric': 'MultiRMSE', 
          'task_type': 'CPU',   # FIXME: GPU is not supported for multirmse
          'iterations': 200,
          'od_type': 'Iter', 
          'boosting_type': 'Plain', 
          'bootstrap_type': 'Bayesian', 
          'allow_const_label': True, 
          'random_state': 1
         }
model = CatBoostRegressor(**params)

n = 1
np.random.seed(42)
all_row_indices = np.arange(train_X.shape[0])
np.random.shuffle(all_row_indices)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

index = 0
score = []

# model = Ridge(copy_X=False)
d = train_X.shape[0]//n
for i in range(0, n*d, d):
    print(f'start [{i}:{i+d}]')
    ind = all_row_indices[i:i+d]    
    for idx_tr, idx_va in kf.split(ind):
        X = train_X[ind]
        Y = train_target[ind] #.todense()
        Yva = train_y[ind][idx_va]
        Xtr, Xva = X[idx_tr], X[idx_va]
        Ytr = Y[idx_tr]
        del X, Y
        gc.collect()
        print('Train...')
        model.fit(Xtr, Ytr)
        del Xtr, Ytr
        gc.collect()
        s = correlation_score(Yva.todense(), model.predict(Xva)@pca2.components_)
        score.append(s)
        print(index, s)
        #del Xva, Yva
        gc.collect()
        pkl_filename = f"model{index:02d}.pkl"
        with open('/workspace/mnt/data1/MSCI/results/240119/catboost/'+pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        gc.collect()
        index += 1
        break
    gc.collect()

#  validation score
mse = mean_squared_error(np.array(Yva.todense()), model.predict(Xva)@pca2.components_)
corrscore = correlation_score(Yva.todense(), model.predict(Xva)@pca2.components_)
"""
MSE: 1.982
Corr: 0.666
"""
# %% Test
test_len = test_X.shape[0]
d = test_len//n
x = []
for i in range(n):
    x.append(test_X[i*d:i*d+d])
# del multi_test_x
gc.collect()

preds = np.zeros((test_len, 23418), dtype='float16')
for i,xx in enumerate(x):
    for ind in range(index):
        print(ind, end=' ')
        with open('/workspace/mnt/data1/MSCI/results/240119/catboost/model00.pkl', 'rb') as file:
            model = pickle.load(file)
        preds[i*d:i*d+d,:] += (model.predict(xx)@pca2.components_)/index
        gc.collect()
    print('')
    del xx
gc.collect()

mse = mean_squared_error(np.array(test_y.todense()), preds)
corrscore = correlation_score(np.array(test_y.todense()), preds)
"""
MSE: 4.597
Corr: nan
>> all zero
"""
# %%
import sys
sys.path.append('/workspace/home/azuma/Personal_Projects/github/ML_DL_Notebook')
from _utils import plot_utils as pu

# %% Feature-Wise
target_idxs = []
original_order = list(targets_idx["columns"])
for ensg in targets_idx['columns']:
    target_idxs.append(original_order.index(ensg))
target_idxs = sorted(target_idxs)

common_all_true = test_rna[:,target_idxs]

pu.plot_scatter(data1=common_all_true.T,data2=preds.T,xlabel="True Value",ylabel="Predicted Value",title="overall",do_plot=True)

#  Each gene
#gene_id = 'ENSG00000104043' # (ATP8B4)
gene_id = 'ENSG00000174059' # CD34
idx = list(targets_idx["columns"]).index(gene_id)

pred_test = preds[:,idx]

rna_array = rna_targets.toarray() # (9300, 23418)
train_rna = rna_array[0:train_size,:]
test_rna = rna_array[train_size:,:]
true_train = list(train_rna[:,idx])
true_test = list(test_rna[:,idx])

pu.plot_scatter(data1=[true_test],data2=[pred_test],xlabel="True Value",ylabel="Predicted Value",title=gene_id+"(CD34): test",do_plot=True)

# %% Sample-Wise
common_idx = []
for i,k in enumerate(targets_idx['columns']):
    if k in original_order:
        common_idx.append(i)
    else:
        pass

#sample_id = '0ce14152aadc' # sample in test
sample_id = '4dc9f351329e' # sample in test
idx = list(targets_idx["index"])[train_size::].index(sample_id)

pred_test = list(preds[idx,:])
true_test = list(test_rna[idx,common_idx])

# for each
pu.plot_scatter(data1=[true_test],data2=[pred_test],xlabel="True Value",ylabel="Predicted Value",title="sample: "+sample_id,do_plot=True)

# %%
#pd.to_pickle(preds,'/workspace/mnt/data1/MSCI/results/240119/catboost/240119_preds.pkl')
