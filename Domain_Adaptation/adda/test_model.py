# -*- coding: utf-8 -*-
"""
Created on 2024-09-27 (Fri) 11:25:58

source.pt: trained on reguar MNIST dataset
adda.pt: adversarial discriminative domain adaptation

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Others'

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(BASE_DIR+'/github/ML_DL_Notebook/Domain_Adaptation/adda')
from data import MNISTM
from models import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path(BASE_DIR+'/datasource')

# %%
batch_size = 64
MODEL_FILE = BASE_DIR+'/github/ML_DL_Notebook/Domain_Adaptation/trained_models/adda.pt'
dataset = MNISTM(train=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=1, pin_memory=True)

model = Net().to(device)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

total_accuracy = 0
with torch.no_grad():
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

mean_accuracy = total_accuracy / len(dataloader)
print(f'Accuracy on target data: {mean_accuracy:.4f}')
