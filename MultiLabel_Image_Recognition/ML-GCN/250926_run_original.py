#!/usr/bin/env python3
"""
Created on 2025-09-26 (Fri) 21:02:11

Running original ML-GCN (https://github.com/megvii-research/ML-GCN) code.

### Modifications
- cuda(async=True)
- comment out @torch.library.register_fake("torchvision::nms") in lib/python3.11/site-packages/torchvision/_meta_registrations.py
- change coco.py (Remove unzip process in coco.py (because I have already unzipped the files)
- change util.py (np.int -> dtype=int)
- change engine.py (self.state['loss'].data[0] -> self.state['loss'].item())
- OutOfMemory Error -> batch size 1
    - This code is executable on inference only.

@author: I.Azuma
"""
# %%
BASE_DIR = "/workspace/cluster/HDD/azuma/Others"
PROJECT_DIR = "/workspace/cluster/HDD/azuma/Others/github/ML-GCN"
DATA_DIR = "/workspace/cluster/HDD/azuma/Others/datasource/COCO2014"
CHECKPOINT_PATH = "/workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ML-GCN/results/coco_checkpoint.pth.tar"

import os
os.chdir(PROJECT_DIR)

# COCO2014
#! python demo_coco_gcn.py {DATA_DIR} --image-size 448 --batch-size 1 -e --resume {CHECKPOINT_PATH} --workers 1

# %%
BASE_DIR = "/workspace/cluster/HDD/azuma/Others"
PROJECT_DIR = "/workspace/cluster/HDD/azuma/Others/github/ML-GCN"
DATA_DIR = "/workspace/cluster/HDD/azuma/Others/datasource/COCO2014"
CHECKPOINT_PATH = "/workspace/cluster/HDD/azuma/Others/github/ML_DL_Notebook/MultiLabel_Image_Recognition/ML-GCN/results/coco_checkpoint.pth.tar"

import os
os.chdir(PROJECT_DIR)

import argparse
from engine import *
from models import *
from coco import *
from util import *

import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=20, type=int, metavar='N')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')

args, unknown = parser.parse_known_args(args=[])

train_dataset = COCO2014(DATA_DIR, phase='train', inp_name='data/coco/coco_glove_word2vec.pkl')

image_normalization_mean = [0.485, 0.456, 0.406]
image_normalization_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=image_normalization_mean,
                                    std=image_normalization_std)
state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':80}
state['train_transform'] = transforms.Compose([
    MultiScaleCrop(state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

train_dataset.transform = state['train_transform']
#train_dataset.target_transform = state['train_target_transform']
train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)

tmp = next(iter(train_loader))

# %%
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

img_tmp = tmp[0][0]  # torch.Size([16, 3, 448, 448])

plt.imshow(img_tmp[3,:,:,:].numpy().transpose(1,2,0))
plt.show()

sns.heatmap(tmp[1])
plt.show()

t=0.4
num_classes=80

adj_file='data/coco/coco_adj.pkl'
result = pickle.load(open(adj_file, 'rb'))
_adj = result['adj']
_nums = result['nums']
_nums = _nums[:, np.newaxis]
_adj = _adj / _nums
_adj[_adj < t] = 0
_adj[_adj >= t] = 1
_adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
_adj = _adj + np.identity(num_classes, dtype=int)
