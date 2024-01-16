# -*- coding: utf-8 -*-
"""
Created on 2023-12-30 (Sat) 17:26:15

Evaluation dev

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
res_df = pd.read_csv('/workspace/mnt/data1/MSCI/results/231230/imputed_exp.csv',index_col=0)
res_df.index = [t.split('_')[0] for t in res_df.index.tolist()]

inputs_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_inputs_idxcol_32606_HSC.npz', allow_pickle=True)

train_size = 6510
samples = inputs_idx['index']
train_samples = samples[0:train_size] # 6510
test_samples = samples[train_size::] # 2790

# targets (rna)
rna_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)

rna_array = rna_targets.toarray() # (9300, 23418)
train_rna = rna_array[0:train_size,:]
test_rna = rna_array[train_size:,:]

# %% eval for each gene
gene_id = 'ENSG00000104043' # (ATP8B4)
idx = list(targets_idx["columns"]).index(gene_id)

pred_train = res_df.loc[train_samples][gene_id].tolist()
pred_test = res_df.loc[test_samples][gene_id].tolist()
true_train = list(train_rna[:,idx])
true_test = list(test_rna[:,idx])

#pu.plot_scatter(data1=[true_train],data2=[pred_train],xlabel="True Value",ylabel="Predicted Value",title=gene_id+"(ATP8B4): train",do_plot=True)
pu.plot_scatter(data1=[true_test],data2=[pred_test],xlabel="True Value",ylabel="Predicted Value",title=gene_id+"(ATP8B4): test",do_plot=True)

# %% overall evaluation
target_idxs = []
original_order = list(targets_idx["columns"])
for ensg in res_df.columns.tolist():
    target_idxs.append(original_order.index(ensg))
target_idxs = sorted(target_idxs)

common_all_true = test_rna[:,target_idxs]

pu.plot_scatter(data1=common_all_true,data2=res_df.loc[test_samples],xlabel="True Value",ylabel="Predicted Value",title="overall",do_plot=True)
