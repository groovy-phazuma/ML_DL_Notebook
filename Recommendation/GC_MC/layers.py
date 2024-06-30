import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from utils import stack, split_stack
from torch_scatter import scatter_mean


# First Layer of the Encoder (implemented by Pytorch Geometric)
# Please the following repository for details.
# https://github.com/rusty1s/pytorch_geometric
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
        
        if config.accum == 'split_stack':
            # each 100 dimention has each realtion node features
            # user-item-weight-sharing
            self.base_weight = nn.Parameter(torch.Tensor(
                max(self.num_users, self.num_item), self.out_c))
            self.dropout = nn.Dropout(self.drop_prob)
        else:
            # ordinal basis matrices in_c * out_c = 2625 * 500
            ord_basis = [nn.Parameter(torch.Tensor(1, self.in_c * self.out_c)) for r in range(self.num_relations)]
            self.ord_basis = nn.ParameterList(ord_basis)
        self.relu = nn.ReLU()

        if config.accum == 'stack':
            self.bn = nn.BatchNorm1d(self.in_c * config.num_relations)
        else:
            self.bn = nn.BatchNorm1d(self.in_c)

        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        if self.accum == 'split_stack':
            weight_init(self.base_weight, self.in_c, self.out_c)
        else:
            for basis in self.ord_basis:
                weight_init(basis, self.in_c, self.out_c)

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        return self.propagate(self.accum, edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def propagate(self, aggr, edge_index, x, edge_type, edge_norm, **kwargs):
        assert aggr in ['split_stack', 'stack', 'add', 'mean', 'max']
        edge_index = edge_index.cpu()
        edge_type = edge_type.cpu()

        size = self.in_c
        
        out = self.message(x, edge_type, edge_norm)
        if aggr == 'split_stack':
            out = split_stack(out, edge_index[0], edge_type, dim_size=size)
        elif aggr == 'stack':
            out = stack(out, edge_index[0], edge_type, dim_size=size)
        else:
            #out = torch.scatter(out, edge_index[0])
            # 受信ノードに集約された特徴量を格納するテンソル
            source = edge_index[0] 
            target = edge_index[1]
            aggregated_features = torch.zeros(self.in_c, out.size(1))

            # 送信ノードの特徴量を受信ノードにscatter_addで集約
            aggregated_features = aggregated_features.index_add(0, target, out[source])

            # 受信ノードごとに隣接ノードの数をカウント
            node_degree = torch.zeros(sself.in_c)
            node_degree = node_degree.index_add(0, target, torch.ones(target.size(0)))

            # 平均を取るために集約された特徴量をノードの次数で割る
            out = aggregated_features / node_degree.unsqueeze(1)

        print('layers.py',out.shape)
        out = self.update(out, *update_args)

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
            print('layers.py: weight shape',weight.shape)  # 13125(relation * nodes)x500
            # index has target features index in weitht matrixs
            #index = edge_type * self.in_c + x_j
            index = self.num_relations * self.in_c + x_j
        weight = self.node_dropout(weight)
        out = weight[index]  # (2625, 500)

        # out is edges(160000) x hidden(500)
        #print('out shape:',out.shape)
        #print('edgenorm shape:',edge_norm.shape)

        #return out if edge_norm is None else out * edge_norm.reshape(-1, 1)

        return out

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channles]
        if self.bn:
            aggr_out = self.bn(aggr_out.unsqueeze(0)).squeeze(0)
        if self.relu:
            aggr_out = self.relu(aggr_out)
        return aggr_out

    def node_dropout(self, weight):
        drop_mask = torch.rand(self.in_c) + (1 - self.drop_prob)
        drop_mask = torch.floor(drop_mask).type(torch.float)
        if self.accum == 'split_stack':
            drop_mask = drop_mask.unsqueeze(1)
        else:
            drop_mask = torch.cat(
                [drop_mask for r in range(self.num_relations)],
                dim=0,
            ).unsqueeze(1)

        drop_mask = drop_mask.expand(drop_mask.size(0), self.out_c)

        assert weight.shape == drop_mask.shape
        
        weight = weight.to(self.device)
        drop_mask = drop_mask.to(self.device)

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
        if config.accum == 'stack':
            self.bn_u = nn.BatchNorm1d(config.num_users * config.num_relations)
            self.bn_i = nn.BatchNorm1d((
                config.num_nodes - config.num_users) * config.num_relations
            )
        else:
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
