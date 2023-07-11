# -*- coding: utf-8 -*-
"""
Created on 2023-06-13 (Tue) 22:39:37

BA-2motifs dataset from Parameterized Explainer for Graph Neural Network

@author: I.Azuma
"""

#%%
import torch
print(torch.__version__) # 2.0.1
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 3090

from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch_geometric.data import Data,DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
from matplotlib import pyplot as plt

dataset1 = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=25, num_edges=1),
    motif_generator=HouseMotif(),
    num_motifs=5,
    num_graphs=1000,
)

dataset2 = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=25, num_edges=1),
    motif_generator=CycleMotif(5),
    num_motifs=5,
    num_graphs=1000,
)

dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

#%%
data_size = len(dataset) # 2000
data_list = []
for idx in range(data_size):
    edge_index = dataset[idx].edge_index # edge index

    n = dataset[idx].num_nodes
    node_feature = [[i,i+1] for i in range(0,n*2,2)]
    x = torch.tensor(node_feature, dtype=torch.float)

    if idx < data_size/2:
        label = 1
    else:
        label = 0
    d = Data(x=x,edge_index=edge_index.contiguous(),t=label)

    data_list.append(d)
    if idx%100 == 99:
        print("\rData loaded "+ str(idx+1), end="  ")

# %%
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 48)
        self.conv4 = GCNConv(48, 64)
        self.conv5 = GCNConv(64, 96)
        self.linear1 = torch.nn.Linear(96,64)
        self.linear2 = torch.nn.Linear(64,10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

#%%
train_size = 1000
batch_size = 64
epoch_num = 150

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

model = Net().to(DEVICE)
trainset = data_list[:train_size]
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = data_list[train_size:]
testloader = DataLoader(testset, batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
history = {
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
}

print("Start Train")

# train phase
model.train()
for epoch in range(epoch_num):
    train_loss = 0.0
    for i, batch in enumerate(trainloader):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs,batch.t)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.cpu().item()
        if i % 10 == 9:
            progress_bar = '['+('='*((i+1)//10))+(' '*((train_size//100-(i+1))//10))+']'
            print('\repoch: {:d} loss: {:.3f}  {}'
                    .format(epoch + 1, loss.cpu().item(), progress_bar), end="  ")

    print('\repoch: {:d} loss: {:.3f}'
        .format(epoch + 1, train_loss / (train_size / batch_size)), end="  ")
    history["train_loss"].append(train_loss / (train_size / batch_size))

    correct = 0
    total = 0
    batch_num = 0
    loss = 0
    with torch.no_grad():
        for data in testloader:
            data = data.to(DEVICE)
            outputs = model(data)
            loss += criterion(outputs,data.t)
            _, predicted = torch.max(outputs, 1)
            total += data.t.size(0)
            batch_num += 1
            correct += (predicted == data.t).sum().cpu().item()

    history["test_acc"].append(correct/total)
    history["test_loss"].append(loss.cpu().item()/batch_num)
    endstr = ' '*max(1,(train_size//1000-39))+"\n"
    print('Test Accuracy: {:.2f} %%'.format(100 * float(correct/total)), end='  ')
    print(f'Test Loss: {loss.cpu().item()/batch_num:.3f}',end=endstr)


print('Finished Training')

# %%
