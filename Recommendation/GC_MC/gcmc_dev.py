# -*- coding: utf-8 -*-
"""
Created on 2024-06-22 (Sat) 01:43:57

Graph Convolutional Matrix Completion (GCMC)

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook'

import yaml
import torch
print(torch.cuda.get_device_name())

import sys
sys.path.append(BASE_DIR+'/Recommendation/GC_MC')
import dataset

from utils import calc_rmse, ster_uniform, random_init, init_xavier, init_uniform, Config
from model import GAE
from trainer import Trainer

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset.MCDataset(root='/workspace/mnt/cluster/HDD/azuma/Others/datasource/ml-100k', name='ml-100k')
input_data = data[0]
print(input_data)
input_data = input_data.to(device)

# %%
with open(BASE_DIR+'/Recommendation/GC_MC/config.yml') as f:
    cfg = yaml.safe_load(f)
cfg = Config(cfg)

# add some params to config
cfg.num_nodes = data.num_nodes
cfg.num_relations = data.num_relations
cfg.num_users = int(input_data.num_users)

# set and init model
model = GAE(cfg, random_init).to(device)
model.apply(init_xavier)

# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr, weight_decay=cfg.weight_decay,
)

# train
experiment=False
trainer = Trainer(
    model, data, input_data, calc_rmse, optimizer, experiment,
)
trainer.training(cfg.epochs)

# %%
