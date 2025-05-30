# -*- coding: utf-8 -*-
"""
Created on 2023-12-29 (Fri) 13:53:07

@author: I.Azuma
"""
# %%
import numpy as np
import scipy
import scipy.sparse
from tqdm import tqdm

import mudata
from mudata import AnnData, MuData

import scanpy as sc

# %%
# 10x
paired = sc.read_h5ad("/workspace/mnt/data1/MSCI/10x_multiome/paired.h5ad")

# kaggle
train_inputs = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_inputs_values_32606_HSC.sparse.npz")
train_inputs = train_inputs.astype('float16', copy=False)
inputs_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_inputs_idxcol_32606_HSC.npz', allow_pickle=True)

# %%
df = paired.var
paired_peak = df[df['modality']=='Peaks']['ID'].tolist()
kaggle_peak = inputs_idx['columns']

# %%
nuc = 'chr1'
paired_nuc = []
for t in paired_peak:
    if t.split(':')[0] == nuc:
        paired_nuc.append(t)
    else:
        pass
paired_nuc = sorted(paired_nuc)

kaggle_nuc = []
for t in kaggle_peak:
    if t.split(':')[0] == nuc:
        kaggle_nuc.append(t)
    else:
        pass
kaggle_nuc = sorted(kaggle_nuc)

# %%
overlap_ids = []
for k in tqdm(kaggle_nuc):
    ks = int(k.split(':')[1].split('-')[0])
    ke = int(k.split(':')[1].split('-')[1])
    tmp_list = []
    for base in paired_nuc:
        s = int(base.split(':')[1].split('-')[0])
        e = int(base.split(':')[1].split('-')[1])
        if s > ke: # over
            break
        elif e < ks: # not yet
            pass
        else:
            tmp_list.append(base)
            break
        
    overlap_ids.append(tmp_list)

"""
{0: 16780, 1:4926}
"""
# %%
import collections

o_len = [len(t) for t in overlap_ids]
c_dic = dict(collections.Counter(o_len))
# %%
