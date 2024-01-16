# -*- coding: utf-8 -*-
"""
Created on 2024-01-05 (Fri) 21:07:40

Link prediction with DistMult
- https://github.com/pyg-team/pytorch_geometric/blob/09733b493493083ab7cb09faf60716308e270093/examples/kge_fb15k_237.py

@author: I.Azuma
"""
# %%
import argparse
import os.path as osp

import torch
import torch.optim as optim

from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

# %%
model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='distmult' ,choices=model_map.keys(), type=str.lower)
#args = parser.parse_args()
args, unknown = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FB15k')

train_data = FB15k_237(path, split='train')[0].to(device)
val_data = FB15k_237(path, split='val')[0].to(device)
test_data = FB15k_237(path, split='test')[0].to(device)

model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args.model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
    **model_arg_map.get(args.model, {}),
).to(device)

loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
)

optimizer_map = {
    'transe': optim.Adam(model.parameters(), lr=0.01),
    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    'rotate': optim.Adam(model.parameters(), lr=1e-3),
}
optimizer = optimizer_map[args.model]

# %%
def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=20000,
        k=10,
    )


for epoch in range(1, 501):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    #if epoch % 25 == 0:
        # rank, mrr, hits = test(val_data) # NOTE: not worked
        # print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, 'f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')


# rank, mrr, hits_at_10 = test(test_data)
#print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, 'f'Test Hits@10: {hits_at_10:.4f}')
a,b = test(test_data)
print(f'Test Loss: {b:.2f}')
"""
100%|██████████| 20466/20466 [00:07<00:00, 2921.53it/s]
Test Loss: 0.39
"""

# %%
