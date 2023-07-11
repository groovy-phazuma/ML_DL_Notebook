# -*- coding: utf-8 -*-
"""
Created on 2023-06-05 (Mon) 12:14:15

MLP development

@author: I.Azuma
"""
#%%
import torch
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

#%% cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
#%%
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

num_batches = 100
train_dataloader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
train_iter = iter(train_dataloader)
imgs, labels = next(train_iter) # FIXME: train_iter.next() doesn't work

# output image
img = imgs[0]
#img_permute = img.permute(1,2,0) # (1,28,28) --> (28,28,1)
#sns.heatmap(img_permute.numpy()[:,:,0])
sns.heatmap(img.numpy()[0,:,:])

#%% train
import sys
sys.path.append('/workspace/github/DL-notebook')
from mlp.model import simple_model

model = simple_model.MLP()
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
losses = []
accs = []
for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    running_acc = 0.0
    for imgs,labels in train_dataloader:
        imgs = imgs.view(num_batches,-1) # (100,784) like converting each figure to a single column
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        running_loss += loss.item()

        pred = torch.argmax(output, dim=1) # return max value
        running_acc += torch.mean(pred.eq(labels).float())
        loss.backward()
        optimizer.step()
    # average
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc)
    print("epoch: {}, loss: {}, acc: {}".format(epoch, running_loss, running_acc))

#%% save the model
