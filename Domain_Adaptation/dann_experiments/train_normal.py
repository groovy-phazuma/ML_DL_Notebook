# -*- coding: utf-8 -*-
"""
Created on 2024-10-12 (Sat) 23:38:16

TODO: 
1. change dataloader (mix two domains)

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook/Domain_Adaptation/dann_experiments'

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
sys.path.append(BASE_DIR)
from da_utils import GrayscaleToRgb
import dann_model

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
""" legacy code for ADDA (not DANN)
DATA_DIR = Path('/workspace/mnt/cluster/HDD/azuma/Others/datasource')
def create_dataloaders(batch_size):
    dataset = MNIST(DATA_DIR/'mnist', train=True, download=True,
                    transform=Compose([GrayscaleToRgb(), ToTensor()]))
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8*len(dataset))]
    val_idx = shuffled_indices[int(0.8*len(dataset)):]

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)
    return train_loader, val_loader
"""


# %%
arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
arg_parser.add_argument('--batch-size', type=int, default=64)
arg_parser.add_argument('--epochs', type=int, default=30)
args, unknown = arg_parser.parse_known_args(args=[])

args.num_epochs = 50


train_loader, val_loader = create_dataloaders(args.batch_size)
# model definition
model = dann_model.DomainAdversarialCNN().to(device)
criterion = model.get_loss_fn()

for x, y_true in tqdm(train_loader, leave=False):
    x, y_true = x.to(device), y_true.to(device)
    preds = model(x, alpha=0.5)
    loss = criterion(
            preds["logits"],
            preds["domain_logits"],
            label,
            domain_label)
    loss = criterion(y_pred, y_true)

    if optim is not None:
        optim.zero_grad()
        loss.backward()
        optim.step()


# %%
