# -*- coding: utf-8 -*-
"""
Created on 2023-12-13 (Wed) 20:40:00

Community detection.

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

# %%
glist, _ = load_graphs(base_dir + '_datasource/cell_graph/consep_test10_cg.bin')
g = glist[0]
type_list = pd.read_pickle(base_dir + '_datasource/cell_graph/consep_test10_type_list.pkl')
type_list = handling.relabel(type_list)

# prepare type set
res = [[] for _ in range(max(type_list)+1)]
for i in range(len(type_list)):
    res[type_list[i]].append(i)

# inner edge
u, v = g.edges() # 4337 edges

inner_edge_idx = []
inter_edge_idx = []
for i in range(len(u)):
    if type_list[u.tolist()[i]] == type_list[v.tolist()[i]]:
        inner_edge_idx.append(i)
    else:
        inter_edge_idx.append(i)

print(f"Inner Edges: {len(inner_edge_idx)}/{g.num_edges()}")
print(f"Inter Edges: {len(inter_edge_idx)}/{g.num_edges()}")

# %%
eids = np.arange(g.num_edges()) # array([   0,    1,    2, ..., 4334, 4335, 4336])

test_size = int(len(eids) * 0.3)
train_size = g.num_edges() - test_size

test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
# %%
