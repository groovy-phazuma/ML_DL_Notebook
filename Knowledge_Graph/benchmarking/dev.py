# -*- coding: utf-8 -*-
"""
Created on 2025-04-21 (Mon) 23:46:19

@author: I.Azuma
"""
# %%
from pykeen.pipeline import pipeline
pipeline_result = pipeline(
    dataset='Nations',
    model='TransE',
)
pipeline_result.save_to_directory('nations_transe')

# %%
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel

# %%
