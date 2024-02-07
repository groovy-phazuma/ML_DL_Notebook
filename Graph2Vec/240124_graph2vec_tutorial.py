# -*- coding: utf-8 -*-
"""
Created on 2024-01-24 (Wed) 22:47:41

conduct tutorial

@author: I.Azuma
"""
# %%
import networkx as nx

import sys
sys.path.append('/workspace/home/azuma/Personal_Projects/github/karateclub')
from karateclub.graph_embedding import Graph2Vec

# %%Graph2Vec attributed example
graphs = []
for i in range(50):
    graph = nx.newman_watts_strogatz_graph(50, 5, 0.3)
    nx.set_node_attributes(graph, {j: str(j) for j in range(50)}, "feature")
    graphs.append(graph)
#model = Graph2Vec(attributed=True)
model = Graph2Vec(use_node_attribute="feature")

model.fit(graphs)
emb = model.get_embedding() # (50, 128)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.clustermap(emb)
plt.show()

emb_corr = pd.DataFrame(emb.T).corr()
sns.heatmap(emb_corr)
plt.show()

# %%
import numpy as np
import umap
import umap.plot

mapper = umap.UMAP(random_state=0,n_neighbors=10, min_dist=0.1, n_components=2)
mapper.fit(emb)

# display with umap.plot
labels = [i for i in range(50)]
umap.plot.points(mapper,labels=np.array(labels))
plt.legend().remove()
