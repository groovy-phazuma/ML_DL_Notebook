# -*- coding: utf-8 -*-
"""
Created on 2023-06-11 (Sun) 22:14:18

karate GNN dev

@author: I.Azuma
"""
#%%
import torch
print(torch.__version__) # 2.0.1
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 3090
import torch.nn.functional as F
 
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
 
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

#%%
def check_graph(data):
    '''Display graph info'''
    print("structure:", data)
    print("key: ", data.keys)
    print("node:", data.num_nodes)
    print("edge:", data.num_edges)
    print("node feature:", data.num_node_features)
    print("isolated node:", data.contains_isolated_nodes())
    print("self loop:", data.contains_self_loops())
    print("=== node feature:x ===")
    print(data['x'])
    print("=== node class:y ===")
    print(data['y'])
    print("=== edge format ===")
    print(data['edge_index']) # sender and receiver

# load dataset
dataset = KarateClub()
 
print("graph number:", len(dataset))
print("class number:",dataset.num_classes) 
 
data = dataset[0]
check_graph(data)

#%%
nxg = to_networkx(data)
 
# pagerank
pr = nx.pagerank(nxg)
pr_max = np.array(list(pr.values())).max()
 
# layout
draw_pos = nx.spring_layout(nxg, seed=0) 
 
# node color
cmap = plt.get_cmap('tab10')
labels = data.y.numpy()
colors = [cmap(l) for l in labels]
 
# display
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(nxg, 
                       draw_pos,
                       node_size=[v / pr_max * 1000 for v in pr.values()],
                       node_color=colors, alpha=0.5)
nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
nx.draw_networkx_labels(nxg, draw_pos, font_size=10)
 
plt.title('KarateClub')
plt.show()

#%%
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_size = 10
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, dataset.num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

#%% train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
model.train() # training phase
# input data
data = dataset[0]
 
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# learnig loop
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

model.eval() # evaluation phase
_, pred = model(data).max(dim=1)

print("Results: ", pred)
print("True: ", data["y"])