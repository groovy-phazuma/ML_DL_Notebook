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
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
import torch.nn.functional as F

from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import Compose, ToTensor, Grayscale

import sys
sys.path.append(BASE_DIR)
from da_utils import GrayscaleToRgb
from data import MNISTM
import dann_model

import argparse

# show gpu name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# %%
# legacy code for ADDA (not DANN)
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

# minist dataset
batch_size = 64
mnist_data = MNIST(DATA_DIR/'mnist', train=True, download=True,
                    transform=Compose([GrayscaleToRgb(), ToTensor()]))
mnistm_data = MNISTM(train=True)

shuffled_indices = np.random.permutation(len(mnist_data))
train_idx = shuffled_indices[:int(0.8*len(mnist_data))]
val_idx = shuffled_indices[int(0.8*len(mnist_data)):]
train_loader = DataLoader(mnist_data, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)

# custom dataset
class DomainLabelDataset(Dataset):
    def __init__(self, dataset, domain_label):
        self.dataset = dataset
        self.domain_label = domain_label
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, label = self.dataset[idx]
        return x, label, self.domain_label

custom_mnist = DomainLabelDataset(mnist_data, domain_label=0)
costom_mnistm = DomainLabelDataset(mnistm_data, domain_label=1) 
# combined dataset
combined_dataset = ConcatDataset([custom_mnist, costom_mnistm])

batch_size = 64
train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)

# shuffle dataset and split into train and validation
dataset_size = len(combined_dataset)
indices = np.random.permutation(dataset_size)
split = int(np.floor(0.8 * dataset_size))  # 80% for training
train_indices, val_indices = indices[:split], indices[split:]

# define dataloaders
train_loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))


# %%
arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
arg_parser.add_argument('--epochs', type=int, default=30)
args, unknown = arg_parser.parse_known_args(args=[])

args.num_epochs = 50

# model definition
model = dann_model.DomainAdversarialCNN().to(device)
criterion = model.get_loss_fn()
optim = Adam(model.parameters(), lr=1e-3)

# training loop
trian_loss_history = []
valid_loss_history = []
for epoch in tqdm(range(args.num_epochs)):
    # train
    running_loss_t = 0.0
    model.train()
    for x, label, domain_label in tqdm(train_loader, leave=False):
        x, label, domain_label = x.to(device), label.to(device), domain_label.to(device)
        preds = model(x, alpha=0.5)
        loss_t = criterion(
                preds["logits"],
                preds["domain_logits"],
                label,
                domain_label)

        if optim is not None:
            optim.zero_grad()
            loss_t.backward()
            optim.step()
        
        running_loss_t += loss_t.item()
    trian_loss_history.append(running_loss_t/len(train_loader))
        
    # valid
    running_loss_v = 0.0
    model.eval()
    for x, label, domain_label in tqdm(val_loader, leave=False):
        x, label, domain_label = x.to(device), label.to(device), domain_label.to(device)
        preds = model(x, alpha=0.5)
        loss_v = criterion(
                preds["logits"],
                preds["domain_logits"],
                label,
                domain_label)
        running_loss_v += loss_v.item()
    valid_loss_history.append(running_loss_v/len(val_loader))

# %%
