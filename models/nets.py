import torch.nn as nn
import torch.nn.functional as F
from misc.utils import *


class GCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=64, n_clss=10, args=None, dropout=0.5):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args
        self.dropout = dropout

        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
        self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
        # self.clsif = GCNConv(self.n_dims, self.n_clss, cached=False)
        if self.args.task == 0:
            self.clsif = nn.Linear(self.n_dims, self.n_clss)

    def forward(self, data, adj=None, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if adj is not None:
            edge_index = adj
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        # plot_feature = x
        # x = F.dropout(x, training=self.training)
        # x = self.clsif(x, edge_index, edge_weight)
        if self.args.task == 0:
            x = self.clsif(x)
        else:
            x = x @ x.T
        # print(f'x shape:::::::::::::::::::::::::::::{x.shape}')
        # return x, plot_feature
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.args.task == 0:
            self.clsif.reset_parameters()


class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=64, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        from models.layers import MaskedGCNConv, MaskedLinear
        self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
        # self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
        self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x
