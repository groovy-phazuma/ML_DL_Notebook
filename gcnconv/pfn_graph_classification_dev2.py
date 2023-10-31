# -*- coding: utf-8 -*-
"""
Created on 2023-06-14 (Wed) 22:34:12

PFN dataset
graph classification dev -v2

@author: I.Azuma
"""
#%%
import torch
print(torch.__version__) # 2.0.1
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 3090

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch_geometric.data import Data,DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
from matplotlib import pyplot as plt

import glob
import numpy as np

import sys
sys.path.append('/workspace/github/TopoPathology/dev/gnn_scratch')
from utils import splittraintest
#%%
path = '/workspace/github/TopoPathology/dev/gnn_scratch/datasets/train/'
trainset_pre,testset_pre = splittraintest.split(path,testratio=0.3).gettraintest()

data_size = len(trainset_pre.adj) # 1400
data_list = []
for idx in range(data_size):
    n = trainset_pre.nnodes[idx]
    a = trainset_pre.adj[idx]
    src = []  # sender
    dst = []  # receiver

    for r in range(n):
        indexes = [i for i, x in enumerate(a[r]) if x == 1]
        src.extend([r]*len(indexes))
        dst.extend(indexes)
    # edge
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # node feature
    node_feature = [[i,i+1] for i in range(0,n*2,2)]
    x = torch.tensor(node_feature, dtype=torch.float)

    d = Data(x=x, edge_index=edge_index.contiguous(),t=int(trainset_pre.labels[idx]))
    data_list.append(d)
    if idx%100 == 99:
        print("\rData loaded "+ str(idx+1), end="  ")

# %%
train_size = 1000

train_dataset = data_list[:train_size]
test_dataset = data_list[train_size:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

# %%
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=64)
print(model)

#%%
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

model = GCN(hidden_channels=64)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         data = data.to(DEVICE)
         out = model(data)  # Perform a single forward pass.
         loss = criterion(out, data.t)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(DEVICE)
         out = model(data)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.t).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
# %%
