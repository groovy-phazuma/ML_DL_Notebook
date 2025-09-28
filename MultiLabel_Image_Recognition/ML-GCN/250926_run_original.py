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
! python demo_coco_gcn.py {DATA_DIR} --image-size 448 --batch-size 1 -e --resume {CHECKPOINT_PATH} --workers 1

# %%
