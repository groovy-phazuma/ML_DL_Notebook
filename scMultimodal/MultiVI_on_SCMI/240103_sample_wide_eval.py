# -*- coding: utf-8 -*-
"""
Created on 2024-01-03 (Wed) 20:24:41

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd

import scipy
import scipy.sparse

import sys
sys.path.append('/workspace/home/azuma/Personal_Projects/github/ML_DL_Notebook')
from _utils import plot_utils as pu

# %%
rna_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
preds = pd.read_pickle('/workspace/mnt/data1/MSCI/results/231230/catoost/catboost_preds_2790x23418.pkl')
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)
inputs_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_inputs_idxcol_32606_HSC.npz', allow_pickle=True)

# %% CatBoost
train_size=6510
gene_id = 'ENSG00000104043'
idx = list(targets_idx["columns"]).index(gene_id)

pred_test = preds[:,idx]

rna_array = rna_targets.toarray() # (9300, 23418)
train_rna = rna_array[0:train_size,:]
test_rna = rna_array[train_size:,:]
true_train = list(train_rna[:,idx])
true_test = list(test_rna[:,idx])

pu.plot_scatter(data1=[true_test],data2=[pred_test],xlabel="True Value",ylabel="Predicted Value",title=gene_id+" (ATP8B4)",do_plot=True)

# %% overall evaluation
target_idxs = []
original_order = list(targets_idx["columns"])
for ensg in targets_idx['columns']:
    target_idxs.append(original_order.index(ensg))
target_idxs = sorted(target_idxs)

common_all_true = test_rna[:,target_idxs]

pu.plot_scatter(data1=common_all_true.T,data2=preds.T,xlabel="True Value",ylabel="Predicted Value",title="overall",do_plot=True)
# %%
train_size=6510
#gene_id = 'ENSG00000104043' # atp8b4
gene_id = 'ENSG00000174059' # cd34

rna_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
rna_array = rna_targets.toarray() # (9300, 23418)
train_rna = rna_array[0:train_size,:]
test_rna = rna_array[train_size:,:]

samples = inputs_idx['index']
train_samples = samples[0:train_size] # 6510
test_samples = samples[train_size::] # 2790
muvi_res = pd.read_csv('/workspace/mnt/data1/MSCI/results/231230/imputed_exp.csv',index_col=0)
muvi_res.index = [t.split('_')[0] for t in muvi_res.index.tolist()]
muvi_res = muvi_res.loc[test_samples]
final_genes = muvi_res.columns.tolist()
muvi_array = np.array(muvi_res)
# %%
common_idx = []
for i,k in enumerate(targets_idx['columns']):
    if k in final_genes:
        common_idx.append(i)
    else:
        pass

idx = list(targets_idx["columns"]).index(gene_id)

pred_test = list(muvi_array[:,final_genes.index(gene_id)])
true_test = list(test_rna[:,idx])

# for each
pu.plot_scatter(data1=[true_test],data2=[pred_test],xlabel="True Value",ylabel="Predicted Value",title=gene_id+" (ATP8B4)",do_plot=True)
# %%
