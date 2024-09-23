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
from model import GAE

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
raw_data = dataset.MCDataset(root='/workspace/mnt/cluster/HDD/azuma/Others/datasource/ml-100k', name='ml-100k')
data = raw_data[0]
"""
>> Data(x=[2625], edge_index=[2, 160000], edge_type=[160000], edge_norm=[160000], train_idx=[80000], test_idx=[20000], train_gt=[80000], test_gt=[20000], num_users=[1], num_items=[1])
"""
with open(BASE_DIR+'/Recommendation/MyGCMC/config.yml') as f:
    cfg = yaml.safe_load(f)
cfg = Config(cfg)

# add some params to config
cfg.num_nodes = raw_data.num_nodes
cfg.num_relations = raw_data.num_relations
cfg.num_users = int(data.num_users)

#dat = RGCLayer(config=cfg, weight_init=random_init)
#dat(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type, edge_norm=None)

# %%
model = GAE(cfg, random_init)
model.apply(init_xavier)

# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr, weight_decay=cfg.weight_decay,
)

def calc_rmse(pred, gt):
    pred = F.softmax(pred, dim=1)
    expected_pred = torch.zeros(gt.shape)
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)

    rmse = (gt.to(torch.float) + 1) - expected_pred
    rmse = torch.pow(rmse, 2)
    rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)

    return rmse

# loop
for epoch in range(cfg.epochs):
    # train
    model.train()
    out = model(data.x, data.edge_index, data.edge_type, edge_norm=None)
    loss = F.cross_entropy(out[data.train_idx], data.train_gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_rmse = calc_rmse(out[data.train_idx], data.train_gt)

    # test
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type, edge_norm=None)
    test_rmse = calc_rmse(out[data.test_idx], data.test_gt)

    print('[ Epoch: {:>4}/{} | Loss: {:.6f} | RMSE: {:.6f} | Test RMSE: {:.6f} ]'.format(
                epoch, cfg.epochs, loss, train_rmse, test_rmse))

# %%
rgc_layer = RGCLayer(config=cfg, weight_init=random_init)
out = rgc_layer(x=input_data.x, edge_index=input_data.edge_index, edge_type=input_data.edge_type, edge_norm=None)
print(out)

# %%
