# -*- coding: utf-8 -*-
"""
Created on 2024-01-25 (Thu) 00:25:43

Protein graph classification with Graph2Vec

@author: I.Azuma
"""
# %%
import pandas as pd
import networkx as nx
import numpy as np
import umap
import umap.plot

import seaborn as sns
import matplotlib.pyplot as plt

from torch_geometric.datasets import TUDataset


import sys
sys.path.append('/workspace/home/azuma/Personal_Projects/github/karateclub')
from karateclub.graph_embedding import Graph2Vec
# %%
# Load dataset
dataset = TUDataset('.',name='PROTEINS', use_node_attr=True)

# Collect graphs
graphs = []
labels = []
for data in dataset:
    e_list = []
    tensor_edgelist = data['edge_index']
    for i in range(len(tensor_edgelist[0])):
        e_list.append((int(tensor_edgelist[0][i]), int(tensor_edgelist[1][i])))
    # networkx.classes
    g = nx.from_edgelist(e_list)

    # load features
    x = data['x']
    #nx.set_node_attributes(g, {j: x[j] for j in range(g.number_of_nodes())}, "feature")
    nx.set_node_attributes(g, {j: str(j) for j in range(g.number_of_nodes())}, "feature")

    # Checking the consecutive numeric indexing.
    node_indices = sorted([node for node in g.nodes()])
    numeric_indices = [index for index in range(g.number_of_nodes())]

    if numeric_indices == node_indices:
        graphs.append(g)
        labels.append(int(data["y"]))
    else:
        pass

# %%
model = Graph2Vec(wl_iterations = 2, use_node_attribute="feature", dimensions = 256, 
                  down_sampling = 0.0001, epochs = 100, learning_rate = 0.025, min_count = 10)
model.fit(graphs)
emb = model.get_embedding() # (50, 128)

# Visualize embedding features
sns.clustermap(emb)
plt.show()

# Visualize correlation matrix
emb_corr = pd.DataFrame(emb.T).corr()
sns.heatmap(emb_corr)
plt.show()


# %% UMAP
# Visualize with uma
mapper = umap.UMAP(random_state=0,n_neighbors=10, min_dist=0.5, n_components=2)
mapper.fit(emb)

# Display with umap.plot
umap.plot.points(mapper,labels=np.array(labels))
plt.show()

# %%
