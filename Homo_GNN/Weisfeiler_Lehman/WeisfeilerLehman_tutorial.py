# -*- coding: utf-8 -*-
"""
Created on 2024-01-16 (Tue) 12:27:57

References
- https://ysig.github.io/GraKeL/0.1a8/auto_examples/weisfeiler_lehman_subtree.html#sphx-glr-auto-examples-weisfeiler-lehman-subtree-py

# ! pip install grakel
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram

# Loads the MUTAG dataset
MUTAG = fetch_dataset("MUTAG", verbose=False)
G, y = MUTAG.data, MUTAG.target

>> not worked well.

@author: I.Azuma
"""
# %%
"""
References
- https://github.com/BorgwardtLab/WWL/blob/master/experiments/main.py
"""
import os

import sys
sys.path.append('/workspace/home/azuma/Personal_Projects/github/WWL')
from experiments.wwl import *

dataset = 'MUTAG'
data_path = os.path.join('/workspace/home/azuma/Personal_Projects/github/WWL/data/', dataset)
h=2

#label_sequences = compute_wl_embeddings_discrete(data_path, h)
graph_filenames = retrieve_graph_filenames(data_directory=data_path)
graphs = [ig.read(filename) for filename in graph_filenames]

wl = WeisfeilerLehman()
label_dicts = wl.fit_transform(graphs, h)

# %%
