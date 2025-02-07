# -*- coding: utf-8 -*-
"""
Created on 2023-12-31 (Sun) 16:57:01

Catboost for baseline

@author: I.Azuma
"""
# %%
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

# %%
"""
# sample selection
DATA_DIR = "/mnt/nfs-mnj-archive-02/group/bio/kaggle/single_cell/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")
DONOR_LIST = [32606]
CELL_TYPE_LIST = ['HSC']
DAY_LIST = [2,3,4]

df_meta = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
df_meta = df_meta[df_meta['technology']=='multiome']
cell_id_set = df_meta[((df_meta['donor'].isin(DONOR_LIST)) & (df_meta['cell_type'].isin(CELL_TYPE_LIST)) & (df_meta['day'].isin(DAY_LIST)))].index.tolist()

df = pd.read_pickle('/mnt/nfs-mnj-home-43/i23_azuma/datasource/atac_annotated/train_multi_inputs_gene_mean.pickle')
train_inputs = df.loc[df.index.isin(cell_id_set)]
train_inputs = train_inputs.astype('float16', copy=False)
del df

print('start inputs SVD')
pca = TruncatedSVD(n_components=128, random_state=42)
train_inputs = pca.fit_transform(train_inputs)
print(pca.explained_variance_ratio_.sum())
"""
# %% load data
# inputs (atac)
atac_inputs = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_inputs_values_32606_HSC.sparse.npz")
#atac_inputs = atac_inputs.astype('float16', copy=False)
inputs_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_inputs_idxcol_32606_HSC.npz', allow_pickle=True)

# targets (rna)
rna_targets = scipy.sparse.load_npz("/workspace/mnt/data1/MSCI/processed/train_multi_targets_values_32606_HSC.sparse.npz")
targets_idx = np.load('/workspace/mnt/data1/MSCI/processed/train_multi_targets_idxcol_32606_HSC.npz', allow_pickle=True)

# %% ATAC compression
atac_array = atac_inputs.toarray()
print('start inputs SVD')
pca = TruncatedSVD(n_components=128, random_state=42)
train_inputs = pca.fit_transform(atac_array)
print(pca.explained_variance_ratio_.sum())

# %% RNA compression
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

# %%
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
        with open('/workspace/mnt/data1/MSCI/results/231230/catbost/'+pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        gc.collect()
        index += 1
        break
    gc.collect()
# %% validation score
mse = mean_squared_error(np.array(Yva.todense()), model.predict(Xva)@pca2.components_)
corrscore = correlation_score(Yva.todense(), model.predict(Xva)@pca2.components_)

# %% test
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
        with open('/workspace/mnt/data1/MSCI/results/231230/catboost/model00.pkl', 'rb') as file:
            model = pickle.load(file)
        preds[i*d:i*d+d,:] += (model.predict(xx)@pca2.components_)/index
        gc.collect()
    print('')
    del xx
gc.collect()

pd.to_pickle(preds,'/workspace/mnt/data1/MSCI/results/231230/catoost/catboost_preds_2790x23418.pkl')
# %%
mse = mean_squared_error(np.array(test_y.todense()), preds)
corrscore = correlation_score(np.array(test_y.todense()), preds)
"""
mse: 2.0672932
corr: 0.6750935420576014
"""
# %%
import sys
sys.path.append('/workspace/home/azuma/Personal_Projects/github/ML_DL_Notebook')
from _utils import plot_utils as pu

# %% eval for each gene
gene_id = 'ENSG00000104043' # (ATP8B4)
idx = list(targets_idx["columns"]).index(gene_id)

pred_test = preds[:,idx]

rna_array = rna_targets.toarray() # (9300, 23418)
train_rna = rna_array[0:train_size,:]
test_rna = rna_array[train_size:,:]
true_train = list(train_rna[:,idx])
true_test = list(test_rna[:,idx])

pu.plot_scatter(data1=[true_test],data2=[pred_test],xlabel="True Value",ylabel="Predicted Value",title=gene_id+"(ATP8B4): test",do_plot=True)
# %%
