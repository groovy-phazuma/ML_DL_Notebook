# -*- coding: utf-8 -*-
"""
Created on 2024-01-06 (Sat) 20:25:11

Learn nn.Parameter class behavior.

References
- https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
- https://take-tech-engineer.com/pytorch-parameters-init/

@author: I.Azuma
"""
# %%
import torch
import torch.nn as nn

# %%
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)     
        return x

model = Model()
print(model)
# Model(
#   (linear): Linear(in_features=3, out_features=3, bias=True)
#   (relu): ReLU()
# )

## initial values for weight and bias
for param in model.parameters():
  print(param)

# Parameter containing:
# tensor([[ 0.2242, -0.3569,  0.1490],
#         [-0.2229, -0.1892,  0.3646],
#         [-0.1613, -0.3112,  0.4168]], requires_grad=True)
# Parameter containing:
# tensor([-0.3672, -0.3400, -0.2994], requires_grad=True)

# %%
model = Model()

weight_new = nn.Parameter(torch.tensor([[1., 2., 3.],
                                          [4., 5., 6.],
                                          [7., 8., 9.]])) 
bias_new = nn.Parameter(torch.tensor([[10., 11., 12.]])) 

model.linear.weight = weight_new
model.linear.bias = bias_new

for param in model.parameters():
  print(param)

# %%