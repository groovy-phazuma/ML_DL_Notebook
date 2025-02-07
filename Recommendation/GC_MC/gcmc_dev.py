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
"""
'/datasource/ml-100k/raw/u1.base',
 '/datasource/ml-100k/raw/u1.test'
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = dataset.MCDataset(root='/workspace/mnt/cluster/HDD/azuma/Others/datasource/ml-100k', name='ml-100k')
#data.process()
input_data = data[0]
print(input_data)

# %%
with open(BASE_DIR+'/Recommendation/GC_MC/config.yml') as f:
    cfg = yaml.safe_load(f)
cfg = Config(cfg)

# add some params to config
cfg.num_nodes = data.num_nodes
cfg.num_relations = data.num_relations
cfg.num_users = int(input_data.num_users)

# set and init model
model = GAE(cfg, random_init)
model.apply(init_xavier)

# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr, weight_decay=cfg.weight_decay,
)

input_data = input_data.to(cfg.device)
model = model.to(cfg.device)
# %% Train
experiment=False
trainer = Trainer(
    model=model, data=input_data, calc_rmse=calc_rmse, optimizer=optimizer, experiment=experiment)
trainer.training(cfg.epochs)

trainer.model.train()

# %%
import pandas as pd

csv_path = '/workspace/mnt/cluster/HDD/azuma/Others/datasource/ml-100k/raw/u1.base'
col_names = ['user_id', 'item_id', 'relation', 'ts']
df = pd.read_csv(csv_path, sep='\t', names=col_names)
df = df.drop('ts', axis=1)
df['user_id'] = df['user_id'] - 1
df['item_id'] = df['item_id'] - 1
df['relation'] = df['relation'] - 1

nums = {'user': df.max()['user_id'] + 1,
        'item': df.max()['item_id'] + 1,
        'node': df.max()['user_id'] + df.max()['item_id'] + 2,
        'edge': len(df)}
# >> {'user': 943, 'item': 1682, 'node': 2625, 'edge': 80000}
df['item_id'] = df['item_id'] + nums['user']
x = torch.arange(nums['node'], dtype=torch.long)

# %%
model.train()
out = model(input_data.x, input_data.edge_index, input_data.edge_type, input_data.edge_norm)
loss = F.cross_entropy(out[input_data.train_idx], input_data.train_gt)

optimizer.zero_grad()
loss.backward()
optimizer.step()

rmse = self.calc_rmse(out[self.data.train_idx], self.data.train_gt)
return loss.item(), rmse.item()

# %%
