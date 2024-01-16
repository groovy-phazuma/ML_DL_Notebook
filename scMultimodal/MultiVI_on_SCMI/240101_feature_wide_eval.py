# -*- coding: utf-8 -*-
"""
Created on 2024-01-01 (Mon) 18:54:40

feature wide evaluation (for each sample)

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

# %% True values
train_size=6510
rna_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)

rna_array = rna_targets.toarray() # (9300, 23418)
train_rna = rna_array[0:train_size,:]
test_rna = rna_array[train_size:,:]

# %% catboost prediction
cat_res = pd.read_pickle('/workspace/mnt/data1/MSCI/results/231230/catoost/catboost_preds_2790x23418.pkl')

# %% eval for each sample
sample_id = '0ce14152aadc' # sample in test
idx = list(targets_idx["index"])[train_size::].index(sample_id)

pred_test = list(cat_res[idx,:])
true_test = list(test_rna[idx,:])

pu.plot_scatter(data1=[true_test],data2=[pred_test],xlabel="True Value",ylabel="Predicted Value",title="sample: "+sample_id,do_plot=True)

# %% eval for overall samples
pu.plot_scatter(data1=test_rna,data2=cat_res,xlabel="True Value",ylabel="Predicted Value",title="overall",do_plot=True)

# %% multivi prediction
train_size = 6510
inputs_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_inputs_idxcol_32606_HSC.npz', allow_pickle=True)
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)

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

sample_id = '0ce14152aadc' # sample in test
idx = list(targets_idx["index"])[train_size::].index(sample_id)

pred_test = list(muvi_array[idx,:])
true_test = list(test_rna[idx,common_idx])

# for each
pu.plot_scatter(data1=[true_test],data2=[pred_test],xlabel="True Value",ylabel="Predicted Value",title="sample: "+sample_id,do_plot=True)

# overall
pu.plot_scatter(data1=test_rna[:,common_idx],data2=muvi_array,xlabel="True Value",ylabel="Predicted Value",title="overall",do_plot=True)
# %%
