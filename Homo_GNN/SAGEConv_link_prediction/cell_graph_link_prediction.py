# -*- coding: utf-8 -*-
"""
Created on 2023-12-08 (Fri) 11:38:31

link prediction with ce-graph

@author: I.Azuma
"""
# %%
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data.utils import load_graphs

base_dir = '/workspace/home/azuma/Personal_Projects/github/ML_DL_Notebook/'
import sys
sys.path.append(base_dir)
from Homo_GNN._utils import handling

# %% Load dataset
glist, _ = load_graphs(base_dir + '_datasource/cell_graph/consep_test10_cg.bin')
g = glist[0]
type_list = pd.read_pickle(base_dir + '_datasource/cell_graph/consep_test10_type_list.pkl')
type_list = handling.relabel(type_list,start=1)
print(set(type_list))


u, v = g.edges() # 4337 edges
edge_idx = [[] for _ in range(max(type_list)+1)]
for i in range(len(u)):
    u_type =  type_list[u.tolist()[i]]
    v_type = type_list[v.tolist()[i]]
    if u_type == v_type:
        edge_idx[u_type].append(i)
    else:
        edge_idx[0].append(i)

for i in range(len(edge_idx)):
    if i == 0:
        print(f'Inter edges: {len(edge_idx[0])}')
    else:
        print(f'Inner between {i} edges: {len(edge_idx[i])}')



# %% Prepare training and test sets
u, v = g.edges() # 4337 edges

eids = np.arange(g.num_edges()) # array([   0,    1,    2, ..., 4334, 4335, 4336])
eids = np.random.permutation(eids)

test_size = int(len(eids) * 0.1)
train_size = g.num_edges() - test_size

test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
neg_u, neg_v = np.where(adj_neg != 0) # 797583 edges

neg_eids = np.random.choice(len(neg_u), g.num_edges())
test_neg_u, test_neg_v = (
    neg_u[neg_eids[:test_size]],
    neg_v[neg_eids[:test_size]],
)
train_neg_u, train_neg_v = (
    neg_u[neg_eids[test_size:]],
    neg_v[neg_eids[test_size:]],
)

# When training, you will need to remove the edges in the test set from the original graph
train_g = dgl.remove_edges(g, eids[:test_size])

# %%
from dgl.nn import SAGEConv

# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
# %%
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
    
model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
pred = MLPPredictor(16)

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)
# %%
# in this case, loss will in training loop
optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=0.01
)

# training
all_logits = []
for e in range(100):
    # forward
    h = model(train_g, train_g.ndata["feat"])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# check the results
from sklearn.metrics import roc_auc_score

with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))

# %%
