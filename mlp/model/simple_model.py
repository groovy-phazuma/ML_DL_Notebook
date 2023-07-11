# -*- coding: utf-8 -*-
"""
Created on 2023-06-05 (Mon) 11:07:23

MLP with Pytorch

@author: I.Azuma
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%% simple
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28,400),
            nn.ReLU(inplace=True),
            nn.Linear(400,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,10)
        )
    
    def forward(self,x):
        output = self.classifier(x)
        return output