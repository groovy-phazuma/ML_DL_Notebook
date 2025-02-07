# -*- coding: utf-8 -*-
"""
Created on 2024-10-13 (Sun) 00:25:31

@author: I.Azuma
"""
# %%
import torch
import torch.nn as nn

from torch.autograd import Function

# %%
class GradientReversalLayer(Function):
    @staticmethod
    def forward(context, x, constant):
        context.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(context, grad):
        return grad.neg() * context.constant, None


class Attention(nn.Module):
    def __init__(self,
                 feature_dims: int,
                 step_dims: int,
                 n_middle: int,
                 n_attention: int,
                 **kwargs):
        super().__init__()
        self.support_masking = True
        self.feature_dims = feature_dims
        self.step_dims = step_dims
        self.n_middle = n_middle
        self.n_attention = n_attention
        self.feature_dims = 0

        self.lin1 = nn.Linear(feature_dims, n_middle, bias=False)
        self.lin2 = nn.Linear(n_middle, n_attention, bias=False)

    def forward(self, x, mask=None):
        step_dims = self.step_dims

        eij = self.lin1(x)
        eij = torch.tanh(eij)
        eij = self.lin2(eij)

        a = torch.exp(eij).reshape(-1, self.n_attention, step_dims)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 2, keepdims=True) + 1e-10

        weighted_input = torch.bmm(a, x)
        return torch.sum(weighted_input, 1)

# %%
class DomainAdversarialCNN(nn.Module):
    def __init__(self, img_size=28, warmup=True):
        super().__init__()
        self.img_size = img_size
        self.warmup = warmup
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )

        in_features = self._get_in_features()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def _get_in_features(self):
        in_channels = self.feature_extractor[0].in_channels
        dummy = torch.ones((1, in_channels, self.img_size, self.img_size))
        out = self.feature_extractor(dummy)
        return out.size(1) * (out.size(2) ** 2)

    def forward(self, x, alpha):
        batch_size = x.size(0)
        x = self.feature_extractor(x).view(batch_size, -1) 
        if self.warmup:
            y = GradientReversalLayer.apply(x, alpha)
        else:
            y = GradientReversalLayer.apply(x, 1.0)
        x = self.classifier(x)
        y = self.domain_classifier(y)
        return {
            "logits": x.view(batch_size, -1),
            "domain_logits": y.view(batch_size, -1)
        }

    def get_loss_fn(self):
        class DANNLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss_classifier = nn.CrossEntropyLoss()
                self.loss_domain = nn.BCEWithLogitsLoss()

            def forward(self, x, y, target, domain_target):
                source_preds = x[domain_target == 0]
                source_target = target[domain_target == 0]

                source_classification_loss = self.loss_classifier(
                    source_preds, source_target)

                domain_classification_loss = self.loss_domain(
                    y.view(-1), domain_target.float())
                return source_classification_loss + domain_classification_loss
        return DANNLoss()


class NaiveClassificationCNN(nn.Module):
    def __init__(self, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )

        in_features = self._get_in_features()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def _get_in_features(self):
        in_channels = self.feature_extractor[0].in_channels
        dummy = torch.ones((1, in_channels, self.img_size, self.img_size))
        out = self.feature_extractor(dummy)
        return out.size(1) * (out.size(2) ** 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x).view(batch_size, -1)
        x = self.classifier(x)
        return {
            "logits": x.view(batch_size, -1)
        }

    def get_loss_fn(self):
        class NaiveClassificationLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss_classifier = nn.CrossEntropyLoss()

            def forward(self, x, target, domain_target):
                source_preds = x[domain_target == 0]
                source_target = target[domain_target == 0]

                source_classification_loss = self.loss_classifier(
                    source_preds, source_target)

                return source_classification_loss
        return NaiveClassificationLoss()
