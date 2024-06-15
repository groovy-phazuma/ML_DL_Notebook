# -*- coding: utf-8 -*-
"""
Created on 2024-06-15 (Sat) 20:34:05

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Others'

import numpy as np

import sys
sys.path.append(BASE_DIR+'/github/ML_DL_Notebook_Recommendation')
import recomm_utils as ru

# %%
path_dataset = '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook/_datasource/yahoo_music/training_test_dataset.mat'

ru.load_data(path_dataset= '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook/_datasource/yahoo_music/training_test_dataset.mat',seed=1234, verbose=True)
# %%
