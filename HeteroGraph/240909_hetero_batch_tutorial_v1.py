# -*- coding: utf-8 -*-
"""
Created on 2024-09-09 (Mon) 21:13:55

mini-batch training for heterogenous graph data

Reference:
- https://docs.dgl.ai/en/0.8.x/guide/minibatch.html


@author: I.Azuma
"""
# %%
import numpy as np

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

# load graph information
dataset = dgl.data.CiteseerGraphDataset()
g = dataset[0]

num_nodes = g.num_nodes()

# split train, validation, test
train_ratio = 0.8
val_ratio = 0.1

nids = np.arange(num_nodes)
np.random.shuffle(nids)

train_size = int(num_nodes * train_ratio)
val_size = int(num_nodes * val_ratio)

train_nids = torch.tensor(nids[:train_size])
val_nids = torch.tensor(nids[train_size:train_size+val_size])
test_nids = torch.tensor(nids[train_size+val_size:])

# data loader
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
dataloader = dgl.dataloading.DataLoader(
    g, train_nids, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)  # num_workers=1 will be faster than 0
input_nodes, output_nodes, blocks = next(iter(dataloader))
print(blocks)

# %%
# 

