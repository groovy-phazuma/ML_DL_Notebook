#!/usr/bin/env python3
"""
Created on 2025-10-08 (Wed) 11:22:21

Mattia Rigotti, Christoph Miksovic, Ioana Giurgiu, Thomas Gschwind, Paolo Scotton, "Attention-based Interpretability with Concept Transformers", in International Conference on Learning Representations (ICLR), 2022 [OpenReview]

# Modifications
- amp=True --> False
- from torchmetrics.functional import accuracy


@author: I.Azuma
"""
# %%
PROJECT_DIR = "/workspace/cluster/HDD/azuma/Others/github/concept_transformer"

import os
os.chdir(PROJECT_DIR)

#! python ctc_mnist.py --learning_rate 0.0004 --max_epochs 150 --warmup 20 --batch_size 32 --n_train_samples 200 --expl_lambda 2.0

# %%
PROJECT_DIR = "/workspace/cluster/HDD/azuma/Others/github/concept_transformer"
import os
os.chdir(PROJECT_DIR)

import torch
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt

import sys
sys.path.append(PROJECT_DIR)
from ctc import load_exp
from viz_utils import batch_predict_results, plot_explanation

# Load checkpoint of trained model
model, data_module = load_exp('./mnist_ctc/ExplanationMNIST_expl5.0/')
results =  batch_predict_results(Trainer().predict(model, data_module))

def plot_prediction(results, idx):
    """Plots prediction, concept attention scores and ground truth
        explanationfor correct predictions
    """
    img = data_module.mnist_test[idx][0].squeeze()

    predict_labs = {0: 'even', 1: 'odd'}
    correct_labs = {0: 'wrong', 1: 'correct'}

    predict = predict_labs[results['preds'][idx].item()]
    correct = correct_labs[results['correct'][idx].item()]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(1,-1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(1,-1), ax3)
    ax3.set_title('concept attention scores')

    return fig

def plot_wrong_prediction(results, num):
    """Plots prediction, concept attention scores and ground truth
        explanationfor incorrect predictions
    """
    errors_ind = torch.nonzero(results['correct'] == 0)

    idx = errors_ind[num].item()
    img = data_module.mnist_test[idx][0].squeeze()

    predict_labs = {0: 'even', 1: 'odd'}
    correct_labs = {0: 'wrong', 1: 'correct'}

    predict = predict_labs[results['preds'][idx].item()]
    correct = correct_labs[results['correct'][idx].item()]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(1,-1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(1,-1), ax3)
    ax3.set_title('concept attention scores')

    return fig

fig = plot_prediction(results, 10)
fig = plot_wrong_prediction(results, 0)

# %%
