# -*- coding: utf-8 -*-
"""
Created on 2024-07-02 (Tue) 21:38:20

@author: I.Azuma
"""
# %%
import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# %%
class RGCLayer(MessagePassing):
    def __init__(self, config, weight_init):
        super(RGCLayer, self).__init__()

        self.in_c = config.num_nodes
        self.out_c = config.hidden_size[0]
        self.num_relations = config.num_relations
        self.num_users = config.num_users
        self.num_item = config.num_nodes - config.num_users
        self.drop_prob = config.drop_prob
        self.weight_init = weight_init
        self.accum = config.accum
        self.bn = config.rgc_bn
        self.relu = config.rgc_relu
        self.device = config.device

        ord_basis = [nn.Parameter(torch.Tensor(1, self.in_c * self.out_c)) for r in range(self.num_relations)]
        self.ord_basis = nn.ParameterList(ord_basis)

        self.reset_parameters()

        if config.accum == 'split_stack':
            # each 100 dimention has each realtion node features
            # user-item-weight-sharing
            self.base_weight = nn.Parameter(torch.Tensor(
                max(self.num_users, self.num_item), self.out_c))
            self.dropout = nn.Dropout(self.drop_prob)
        else:
            # ordinal basis matrices in_c * out_c = 2625 * 500
            ord_basis = [nn.Parameter(torch.Tensor(1, in_c * out_c)) for r in range(self.num_relations)]
            self.ord_basis = nn.ParameterList(ord_basis)

    def reset_parameters(self):
        for basis in self.ord_basis:
            self.weight_init(basis, self.in_c, self.out_c)
    
    def forward(self, x, edge_index, edge_type, edge_norm=None):
        features = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

        return features

    def propagate(self, edge_index, x, edge_type, edge_norm, **kwargs):

        edge_index = edge_index.cpu()
        edge_type = edge_type.cpu()

        out = self.message(x_j=x, edge_type=edge_type, edge_norm=edge_norm)

        return out

    def message(self, x_j, edge_type, edge_norm):
        # create weight using ordinal weight sharing
        if self.accum == 'split_stack':
            weight = torch.cat((self.base_weight[:self.num_users],
                self.base_weight[:self.num_item]), 0)
            # weight = self.dropout(weight)
            index = x_j
            
        else:
            for relation in range(self.num_relations):
                if relation == 0:
                    weight = self.ord_basis[relation]
                else:
                    weight = torch.cat((weight, weight[-1] 
                        + self.ord_basis[relation]), 0)

            # weight (R x (in_dim * out_dim)) reshape to (R * in_dim) x out_dim
            # weight has all nodes features
            weight = weight.reshape(-1, self.out_c)
            # index has target features index in weight matrix
            index = edge_type * self.in_c + x_j
            # this opration is that index(160000) specify the nodes idx in weight matrix
            # for getting the features corresponding edge_index

        #weight = self.node_dropout(weight)
        out = weight[index]

        # out is edges(160000) x hidden(500)
        return out if edge_norm is None else out * edge_norm.reshape(-1, 1)
    
    def node_dropout(self, weight):
        drop_mask = torch.rand(self.in_c) + (1 - self.drop_prob)
        drop_mask = torch.floor(drop_mask).type(torch.float)

        drop_mask = torch.cat(
            [drop_mask for r in range(self.num_relations)],
            dim=0,
        ).unsqueeze(1)
        drop_mask = drop_mask.expand(drop_mask.size(0), self.out_c)

        assert weight.shape == drop_mask.shape, f'{weight.shape} != {drop_mask.shape}'
        
        weight = weight.cpu()
        drop_mask = drop_mask.cpu()

        weight = weight * drop_mask

        return weight

# Second Layer of the Encoder
class DenseLayer(nn.Module):
    def __init__(self, config, weight_init, bias=False):
        super(DenseLayer, self).__init__()
        in_c = config.hidden_size[0]
        out_c = config.hidden_size[1]
        self.bn = config.dense_bn
        self.relu = config.dense_relu
        self.weight_init = weight_init

        self.dropout = nn.Dropout(config.drop_prob)
        self.fc = nn.Linear(in_c, out_c, bias=bias)

        self.bn_u = nn.BatchNorm1d(config.num_users)
        self.bn_i = nn.BatchNorm1d(config.num_nodes - config.num_users)
        self.relu = nn.ReLU()

    def forward(self, u_features, i_features):
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)
        if self.bn:
            u_features = self.bn_u(
                    u_features.unsqueeze(0)).squeeze()
        if self.relu:
            u_features = self.relu(u_features)

        i_features = self.dropout(i_features)
        i_features = self.fc(i_features)
        if self.bn:
            i_features = self.bn_i(
                    i_features.unsqueeze(0)).squeeze()
        if self.relu:
            i_features = self.relu(i_features)

        return u_features, i_features

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
