import torch
import torch.nn as nn
from layers import RGCLayer, DenseLayer

# Main Model
class GAE(nn.Module):
    def __init__(self, config, weight_init):
        super(GAE, self).__init__()
        self.gcenc = GCEncoder(config, weight_init)
        self.bidec = BiDecoder(config, weight_init)

    def forward(self, x, edge_index, edge_type, edge_norm):
        u_features, i_features = self.gcenc(x, edge_index, edge_type, edge_norm)
        #adj_matrices = self.bidec(u_features, i_features)

        return adj_matrices


# Encoder (RGCLayer + DenseLayer)
class GCEncoder(nn.Module):
    def __init__(self, config, weight_init):
        super(GCEncoder, self).__init__()
        self.num_relations = config.num_relations
        self.num_users = config.num_users
        self.accum = config.accum

        self.rgc_layer = RGCLayer(config, weight_init)
        self.dense_layer = DenseLayer(config, weight_init)

    def forward(self, x, edge_index, edge_type, edge_norm):
        # 1. RGCLayer
        features = self.rgc_layer(x, edge_index, edge_type, edge_norm)
        print('passed RGCLayer')
        u_features = features[:self.num_users]
        i_features = features[self.num_users:]

        # 2. DenseLayer
        u_features, i_features = self.dense_layer(u_features, i_features)

        return u_features, i_features



# Decoder
class BiDecoder(nn.Module):
    def __init__(self, config, weight_init):
        super(BiDecoder, self).__init__()
        self.num_basis = config.num_basis
        self.num_relations = config.num_relations
        self.feature_dim = config.hidden_size[1]
        self.accum = config.accum
        self.apply_drop = config.bidec_drop

        self.dropout = nn.Dropout(config.drop_prob)
        self.basis_matrix = nn.Parameter(
                torch.Tensor(config.num_basis, self.feature_dim * self.feature_dim))
        coefs = [nn.Parameter(torch.Tensor(config.num_basis))
                 for b in range(config.num_relations)]
        self.coefs = nn.ParameterList(coefs)

        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        # weight_init(self.basis_matrix, self.feature_dim, self.feature_dim)
        nn.init.orthogonal_(self.basis_matrix)
        for coef in self.coefs:
            weight_init(coef, self.num_basis, self.num_relations)

    def forward(self, u_features, i_features):
        if self.apply_drop:
            u_features = self.dropout(u_features)
            i_features = self.dropout(i_features)
        if self.accum == 'stack':
            u_features = u_features.reshape(self.num_relations, -1, self.feature_dim)
            i_features = i_features.reshape(self.num_relations, -1, self.feature_dim)
            num_users = u_features.shape[1]
            num_items = i_features.shape[1]
        else:
            num_users = u_features.shape[0]
            num_items = i_features.shape[0]
            
        for relation in range(self.num_relations):
            q_matrix = torch.sum(self.coefs[relation].unsqueeze(1) * self.basis_matrix, 0)
            q_matrix = q_matrix.reshape(self.feature_dim, self.feature_dim)
            if self.accum == 'stack':
                if relation == 0:
                    out = torch.chain_matmul(
                            u_features[relation], q_matrix, 
                            i_features[relation].t()).unsqueeze(-1)
                else:
                    out = torch.cat((out, torch.chain_matmul(
                        u_features[relation], q_matrix, 
                        i_features[relation].t()).unsqueeze(-1)),dim=2)
            else:
                if relation == 0:
                    out = torch.chain_matmul(
                            u_features, q_matrix, i_features.t()).unsqueeze(-1)
                else:
                    out = torch.cat((out, torch.chain_matmul(
                        u_features, q_matrix, i_features.t()).unsqueeze(-1)),dim=2)

        out = out.reshape(num_users * num_items, -1)

        return out

