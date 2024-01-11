# -*- coding: utf-8 -*-
"""
Created on 2024-01-11 (Thu) 11:35:15

PC (Constraint-Based Estimator)
- https://pgmpy.org/structure_estimator/pc.html

Dataset: wine
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html

References
- https://qiita.com/JunJun100/items/541103e08a9ea6e23e68

@author: I.Azuma
"""
# %%
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from pgmpy.estimators import PC, HillClimbSearch, BicScore, K2Score, MmhcEstimator, ExhaustiveSearch, TreeSearch 

# %% Prepare input data
# Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names) # Explanatory variable
df['Wine Class'] = wine.target # Objective variable

# Normalization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df =pd.DataFrame(df_scaled,columns=df.columns)

# Feature correlation
corr_matrix = df.corr()
sns.heatmap(corr_matrix,
            square=True,
            cmap='coolwarm',
            xticklabels=corr_matrix.columns.values,
            yticklabels=corr_matrix.columns.values)
plt.show()
fxn = lambda x : abs(x)
abs_adj = corr_matrix.applymap(fxn)

# Correlation with target
corr_y = pd.DataFrame({"features":df.columns,"corr_y":corr_matrix["Wine Class"]},index=None)
corr_y = corr_y.reset_index(drop=True)
corr_y = corr_y.style.background_gradient()

# %% Binning
df_new = df.copy()
for col in wine.feature_names:
    df_new[col] = pd.cut(df_new[col], 5, labels=False)
    
# %%
METHOD = "PC"
METHOD_LIST = ["PC","HillClimb-BS","HillClimb-KS","Tree","Exhaustive"]

# %%
time1 = time.time()
if METHOD == "PC":
    network = PC(df_new)
    best_network = network.estimate()
elif METHODD == "HillClimb-BS":
    network = HillClimbSearch(df_new)
    best_network = network.estimate(scoring_method=BicScore(df_new))
elif METHOD == "HillClimb-KS":
    network = HillClimbSearch(df_new)
    best_network = network.estimate(scoring_method=K2Score(df_new))
elif METHOD == "Tree":
    network = TreeSearch(df_new)
    best_network = network.estimate()
elif METHOD == "Exhaustive":
    network = ExhaustiveSearch(df_new)
    best_network = network.estimate()
elif METHODD == "MMHC":
    network = MmhcEstimator(df)
    best_network = network.estimate()
else:
    raise ValueError("Set Appropriate Method from {}".format(METHOD_LIST))

edge_list = list(best_network.edges())
time2 = time.time()
print(time2-time1)

g = nx.DiGraph()
for e in edge_list:
    src = e[0]
    dst = e[1]
    w = abs_adj[src][dst]
    if w > 0.1:
        g.add_edge(src,dst,weight=w)
    else:
        pass

# display
pos = nx.spring_layout(g, seed=0)
plt.figure(figsize=(10,10)) 
nx.draw_networkx(g, pos) 
plt.show()


nx.write_gml(g,f'/workspace/home/azuma/Personal_Projects/BayesianNetwork/results/{METHOD}_wine_240111.gml')

# %%


