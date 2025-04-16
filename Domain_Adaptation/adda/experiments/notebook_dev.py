# -*- coding: utf-8 -*-
"""
Created on 2025-04-15 (Tue) 13:32:47

Adversarial Domain Adaptation (ADDA)

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
import matplotlib.pyplot as plt

import sys
sys.path.append(BASE_DIR+'/github/ML_DL_Notebook/Domain_Adaptation/adda')
from data import MNISTM
from models import Net
from da_utils import loop_iterable, set_requires_grad, GrayscaleToRgb
import train_source

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path(BASE_DIR+'/datasource')
MODEL_FILE = BASE_DIR+'/github/ML_DL_Notebook/Domain_Adaptation/trained_models/source.pt'

#train_source.create_dataloaders(batch_size=64)

# %% 1. Pre-Training
train_loader, val_loader = train_source.create_dataloaders(batch_size=64)

model = Net().to(device)
optim = torch.optim.Adam(model.parameters())
lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
criterion = torch.nn.CrossEntropyLoss()

epochs = 30

best_accuracy = 0
for epoch in range(1, epochs+1):
    # train
    model.train()
    train_loss, train_accuracy = train_source.do_epoch(model, train_loader, criterion, optim=optim)

    # valid
    model.eval()
    with torch.no_grad():
        val_loss, val_accuracy = train_source.do_epoch(model, val_loader, criterion, optim=None)

    tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

    if val_accuracy > best_accuracy:
        print('Saving model...')
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL_FILE)

    lr_schedule.step(val_loss)

"""                     
EPOCH 030: train_loss=0.0967, train_accuracy=0.9712 val_loss=0.0475, val_accuracy=0.9857
Epoch 00030: reducing learning rate of group 0 to 1.0000e-08.
"""

# %% 2. Adversarial Adaptation (ADDA)
batch_size = 64
# source model
source_model = Net().to(device)
source_model.load_state_dict(torch.load(MODEL_FILE))
source_model.eval()
set_requires_grad(source_model, requires_grad=False)

clf = source_model
source_model = source_model.feature_extractor

# target model
target_model = Net().to(device)
target_model.load_state_dict(torch.load(MODEL_FILE))
target_model = target_model.feature_extractor

discriminator = nn.Sequential(
    nn.Linear(320, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
).to(device)

half_batch = batch_size // 2
source_dataset = MNIST(DATA_DIR/'mnist', train=True, download=True,
                        transform=Compose([GrayscaleToRgb(), ToTensor()]))
source_loader = DataLoader(source_dataset, batch_size=half_batch,
                            shuffle=True, num_workers=1, pin_memory=True)

target_dataset = MNISTM(train=False)
target_loader = DataLoader(target_dataset, batch_size=half_batch,
                            shuffle=True, num_workers=1, pin_memory=True)

discriminator_optim = torch.optim.Adam(discriminator.parameters())
target_optim = torch.optim.Adam(target_model.parameters())
criterion = nn.BCEWithLogitsLoss()

epochs = 5
k_disc = 1
k_clf = 10
iterations = 500
for epoch in range(1, epochs+1):
    batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

    total_loss = 0
    total_accuracy = 0
    for _ in trange(iterations, leave=False):
        # Train discriminator
        set_requires_grad(target_model, requires_grad=False)  # Fix
        set_requires_grad(discriminator, requires_grad=True)
        for _ in range(k_disc):
            (source_x, _), (target_x, _) = next(batch_iterator)
            source_x, target_x = source_x.to(device), target_x.to(device)

            source_features = source_model(source_x).view(source_x.shape[0], -1)
            target_features = target_model(target_x).view(target_x.shape[0], -1)
            
            discriminator_x = torch.cat([source_features, target_features])
            discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                            torch.zeros(target_x.shape[0], device=device)])

            disc_preds = discriminator(discriminator_x).squeeze()
            loss = criterion(disc_preds, discriminator_y)

            discriminator_optim.zero_grad()
            loss.backward()
            discriminator_optim.step()

            total_loss += loss.item()
            total_accuracy += ((disc_preds > 0).long() == discriminator_y.long()).float().mean().item()

        # Train classifier
        set_requires_grad(target_model, requires_grad=True)
        set_requires_grad(discriminator, requires_grad=False)  # Fix
        for _ in range(k_clf):
            _, (target_x, _) = next(batch_iterator)
            target_x = target_x.to(device)
            target_features = target_model(target_x).view(target_x.shape[0], -1)

            # flipped labels
            discriminator_y = torch.ones(target_x.shape[0], device=device)  # not 0 but 1

            clf_preds = discriminator(target_features).squeeze()
            loss = criterion(clf_preds, discriminator_y)

            target_optim.zero_grad()
            loss.backward()
            target_optim.step()

    mean_loss = total_loss / (iterations*k_disc)
    mean_accuracy = total_accuracy / (iterations*k_disc)
    tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                f'discriminator_accuracy={mean_accuracy:.4f}')

    # Create the full target model and save it
    clf.feature_extractor = target_model
    torch.save(clf.state_dict(), BASE_DIR+'/github/ML_DL_Notebook/Domain_Adaptation/trained_models/adda.pt')

"""
EPOCH 001: discriminator_loss=0.3932, discriminator_accuracy=0.8305 
EPOCH 002: discriminator_loss=0.0279, discriminator_accuracy=0.9926     
EPOCH 003: discriminator_loss=0.0158, discriminator_accuracy=0.9953          
EPOCH 004: discriminator_loss=0.0072, discriminator_accuracy=0.9980                               
EPOCH 005: discriminator_loss=0.0073, discriminator_accuracy=0.9980

"""


# %% Testing (model with DA)
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
print(f'Accuracy on target data: {mean_accuracy:.4f}')  # Accuracy on target data: 0.6139

# %% Testing (model with source only)
batch_size = 64
MODEL_FILE = BASE_DIR+'/github/ML_DL_Notebook/Domain_Adaptation/trained_models/source.pt'
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
print(f'Accuracy on target data: {mean_accuracy:.4f}')  # Accuracy on target data: 0.3901


# %%
