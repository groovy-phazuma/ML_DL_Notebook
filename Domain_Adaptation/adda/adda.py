# -*- coding: utf-8 -*-
"""
Created on 2024-09-27 (Fri) 00:31:58

Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)

Reference: https://github.com/jvanvugt/pytorch-domain-adaptation

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Others'

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange
from pathlib import Path

import sys
sys.path.append(BASE_DIR+'/github/ML_DL_Notebook/Domain_Adaptation/adda')
from data import MNISTM
from models import Net
from da_utils import loop_iterable, set_requires_grad, GrayscaleToRgb
import train_source

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path(BASE_DIR+'/datasource')

#train_source.create_dataloaders(batch_size=64)

# %% train source model
"""
train_loader, val_loader = train_source.create_dataloaders(batch_size=64)

model = Net().to(device)
optim = torch.optim.Adam(model.parameters())
lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
criterion = torch.nn.CrossEntropyLoss()

epochs = 30

best_accuracy = 0
for epoch in range(1, epochs+1):
    model.train()
    train_loss, train_accuracy = train_source.do_epoch(model, train_loader, criterion, optim=optim)

    model.eval()
    with torch.no_grad():
        val_loss, val_accuracy = train_source.do_epoch(model, val_loader, criterion, optim=None)

    tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

    if val_accuracy > best_accuracy:
        print('Saving model...')
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), BASE_DIR+'/github/ML_DL_Notebook/Domain_Adaptation/trained_models/source.pt')

    lr_schedule.step(val_loss)

"""

# %%
source_model = Net().to(device)
source_model.load_state_dict(torch.load(args.MODEL_FILE))
source_model.eval()
set_requires_grad(source_model, requires_grad=False)

clf = source_model
source_model = source_model.feature_extractor

target_model = Net().to(device)
target_model.load_state_dict(torch.load(args.MODEL_FILE))
target_model = target_model.feature_extractor

discriminator = nn.Sequential(
    nn.Linear(320, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
).to(device)

half_batch = args.batch_size // 2
source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                        transform=Compose([GrayscaleToRgb(), ToTensor()]))
source_loader = DataLoader(source_dataset, batch_size=half_batch,
                            shuffle=True, num_workers=1, pin_memory=True)

target_dataset = MNISTM(train=False)
target_loader = DataLoader(target_dataset, batch_size=half_batch,
                            shuffle=True, num_workers=1, pin_memory=True)

discriminator_optim = torch.optim.Adam(discriminator.parameters())
target_optim = torch.optim.Adam(target_model.parameters())
criterion = nn.BCEWithLogitsLoss()
