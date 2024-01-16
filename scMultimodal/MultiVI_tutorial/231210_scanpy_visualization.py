# -*- coding: utf-8 -*-
"""
Created on 2023-12-10 (Sun) 22:51:37

scanpy visualization tutorial

References
- https://scanpy-tutorials.readthedocs.io/en/latest/plotting/advanced.html

@author: I.Azuma
"""
# %%
import scanpy as sc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


sc.set_figure_params(figsize=(4, 4), frameon=False)

%config InlineBackend.print_figure_kwargs={"facecolor" : "w"}
%config InlineBackend.figure_format="retina"

# %%

# Inital setting for plot size
from matplotlib import rcParams
FIGSIZE=(3,3)
rcParams['figure.figsize']=FIGSIZE

import warnings
warnings.filterwarnings('ignore')
# %%
adata = sc.datasets.pbmc68k_reduced()
# %%
# Examples of returned objects from the UMAP function
print('Categorical plots:')
axes=sc.pl.umap(adata,color=["bulk_labels"],show=False)
print('Axis from a single category plot:',axes)
plt.close()
axes=sc.pl.umap(adata,color=["bulk_labels",'S_score'],show=False)
print('Axes list from two categorical plots:',axes)
plt.close()
fig=sc.pl.umap(adata,color=["bulk_labels"],return_fig=True)
print('Axes list from a figure with one categorical plot:',fig.axes)
plt.close()

print('\nContinous plots:')
axes=sc.pl.umap(adata,color=["IGJ"],show=False)
print('Axes from one continuous plot:',axes)
plt.close()
fig=sc.pl.umap(adata,color=["IGJ"],return_fig=True)
print('Axes list from a figure of one continous plot:',fig.axes)
plt.close()
# %%
axes=sc.pl.dotplot(adata, ['CD79A', 'MS4A1'], 'bulk_labels', show=False)
print('Axes returned from dotplot object:',axes)
dp=sc.pl.dotplot(adata, ['CD79A', 'MS4A1'], 'bulk_labels', return_fig=True)
print('DotPlot object:',dp)
plt.close()

# %%
# Define matplotlib Axes
# Number of Axes & plot size
ncols=2
nrows=1
figsize=4
wspace=0.5
fig,axs = plt.subplots(nrows=nrows, ncols=ncols,
                       figsize=(ncols*figsize+figsize*wspace*(ncols-1),nrows*figsize))
plt.subplots_adjust(wspace=wspace)
# This produces two Axes objects in a single Figure
print('axes:',axs)

# We can use these Axes objects individually to plot on them
# We need to set show=False so that the Figure is not displayed before we
# finished plotting on all Axes and making all plot adjustments
sc.pl.umap(adata,color='louvain',ax=axs[0],show=False)
# Example zoom-in into a subset of louvain clusters
sc.pl.umap(adata[adata.obs.louvain.isin(['0','3','9']),:],color='S_score',ax=axs[1])
# %%
