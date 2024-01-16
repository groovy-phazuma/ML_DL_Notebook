# -*- coding: utf-8 -*-
"""
Created on 2023-12-28 (Thu) 18:20:03

Load data and process the SCMI dataset.
SCMI data was downloaded from https://www.kaggle.com/competitions/open-problems-multimodal/data.

@author: I.Azuma
"""
# %%
import h5py
import pandas as pd
import scipy
import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

DATA_DIR = "/workspace/mnt/data1/MSCI/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

DONOR_LIST = [32606]
CELL_TYPE_LIST = ['HSC']
DAY_LIST = [2,3,4]

df_meta = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
df_meta = df_meta[df_meta['technology']=='multiome']
cell_id_set = df_meta[((df_meta['donor'].isin(DONOR_LIST)) & (df_meta['cell_type'].isin(CELL_TYPE_LIST)) & (df_meta['day'].isin(DAY_LIST)))].index.tolist()

# %% train inputs
start = 0
chunksize = 5000
total_rows = 0
sparse_chunks_data_list = []
chunks_index_list = []
while True:
    columns_name = None
    multi_train_x = None # Free the memory if necessary
    gc.collect()
    # Read the 5000 rows and select the 30 % subset which is needed for the submission
    multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=start, stop=start+chunksize)
    
    rows_read = len(multi_train_x)
    needed_row_mask = multi_train_x.index.isin(cell_id_set)
    multi_train_x = multi_train_x.loc[needed_row_mask]
    print(multi_train_x.shape)
    if len(multi_train_x) == 0:
        break

    if columns_name is None:
        columns_name = multi_train_x.columns.to_numpy()
    else:
        assert np.all(columns_name == multi_train_x.columns.to_numpy())

    # collect
    chunks_index_list.append(multi_train_x.index.to_numpy())
    chunk_data_as_sparse = scipy.sparse.csr_matrix(multi_train_x.to_numpy())
    sparse_chunks_data_list.append(chunk_data_as_sparse)
    
    # Keep the index (the cell_ids) for later
    multi_train_index = multi_train_x.index
    
    gc.collect()

    if rows_read < chunksize: break # this was the last chunk
    start += chunksize

    print(columns_name)

all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
all_indices = np.hstack(chunks_index_list)

out_filename = '/workspace/mnt/data1/MSCI/processed/'
scipy.sparse.save_npz(out_filename+"train_multi_inputs_values_32606_HSC.sparse", all_data_sparse)
np.savez(out_filename+"train_multi_inputs_idxcol_32606_HSC.npz", index=all_indices, columns =columns_name)

# %% train targets
start = 0
chunksize = 5000
total_rows = 0
sparse_chunks_data_list = []
chunks_index_list = []
while True:
    columns_name = None
    multi_train_x = None # Free the memory if necessary
    gc.collect()
    # Read the 5000 rows and select the 30 % subset which is needed for the submission
    multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=start, stop=start+chunksize)
    
    rows_read = len(multi_train_x)
    needed_row_mask = multi_train_x.index.isin(cell_id_set)
    multi_train_x = multi_train_x.loc[needed_row_mask]
    print(multi_train_x.shape)
    if len(multi_train_x) == 0:
        break

    if columns_name is None:
        columns_name = multi_train_x.columns.to_numpy()
    else:
        assert np.all(columns_name == multi_train_x.columns.to_numpy())

    # collect
    chunks_index_list.append(multi_train_x.index.to_numpy())
    chunk_data_as_sparse = scipy.sparse.csr_matrix(multi_train_x.to_numpy())
    sparse_chunks_data_list.append(chunk_data_as_sparse)
    
    # Keep the index (the cell_ids) for later
    multi_train_index = multi_train_x.index
    
    gc.collect()

    if rows_read < chunksize: break # this was the last chunk
    start += chunksize

all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
all_indices = np.hstack(chunks_index_list)


out_filename = '/workspace/mnt/data1/MSCI/processed/'
scipy.sparse.save_npz(out_filename+"train_multi_targets_values_32606_HSC.sparse", all_data_sparse)
np.savez(out_filename+"train_multi_targets_idxcol_32606_HSC.npz", index=all_indices, columns =columns_name)