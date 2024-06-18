# -*- coding: utf-8 -*-
"""
Created on 2024-06-18 (Tue) 22:35:22

Non-negative matrix factorization

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook/Recommendation'

import numpy as np

import sys
sys.path.append(BASE_DIR)

from MF import nmf
import recomm_utils as ru

# %% Load dataset
path_dataset = '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook/_datasource/yahoo_music/training_test_dataset.mat'

u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values = ru.load_data(path_dataset= '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook/_datasource/yahoo_music/training_test_dataset.mat',seed=1234, verbose=True)

X = rating_mx_train.todense()
mask = np.zeros(X.shape)
mask[u_train_idx,v_train_idx]=1  # binary mask

# %% Non-negative matrix factorization
NMF = nmf.DiegoNMF(num_factors=50,alpha=0.05,tolx=1e-3,
                   max_iter=5000,variance=0.01,verbose=True)
NMF.fix_model(mask=mask, intMat=X)

# %%
