# -*- coding: utf-8 -*-
"""
Created on 2024-07-02 (Tue) 21:12:23

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook'

import yaml
import torch
print(torch.cuda.get_device_name())

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(BASE_DIR+'/Recommendation/MyGCMC')
import dataset
#from model import GCEncoder
from layers import RGCLayer
from utils import calc_rmse, ster_uniform, random_init, init_xavier, init_uniform, Config

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
data = dataset.MCDataset(root='/workspace/mnt/cluster/HDD/azuma/Others/datasource/ml-100k', name='ml-100k')
input_data = data[0]
"""
>> Data(x=[2625], edge_index=[2, 160000], edge_type=[160000], edge_norm=[160000], train_idx=[80000], test_idx=[20000], train_gt=[80000], test_gt=[20000], num_users=[1], num_items=[1])
"""
with open(BASE_DIR+'/Recommendation/MyGCMC/config.yml') as f:
    cfg = yaml.safe_load(f)
cfg = Config(cfg)

# add some params to config
cfg.num_nodes = data.num_nodes
cfg.num_relations = data.num_relations
cfg.num_users = int(input_data.num_users)

# %%
rgc_layer = RGCLayer(config=cfg, weight_init=random_init)
out = rgc_layer(x=input_data.x, edge_index=input_data.edge_index, edge_type=input_data.edge_type, edge_norm=None)
print(out)
# %%
