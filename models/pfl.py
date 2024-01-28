import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN_Net(torch.nn.Module):
    r""" GCN model from the "Semi-supervised Classification with Graph
    Convolutional Networks" paper, in ICLR'17.

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 emb=32,
                 dropout=.0):
        super(GCN_Net, self).__init__()
        # GCNLayer:基础GCN, _H:客户端个性化模型学习
        self.conv_a = GCNLayer(input_dim=in_channels, output_dim=hidden, activation=F.relu, dropout=dropout)
        self.conv_last = GCNLayer(input_dim=hidden, output_dim=out_channels, activation=0, dropout=0)
        self.dropout = dropout
        self.adj = None

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        self.adj = None

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        if self.adj is None:
            self.adj = self.preprocess_graph_new(edge_index, x.size(0))
        x = self.conv_a(x, self.adj)
        x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        x = self.conv_last(x, self.adj)
        return x

    @staticmethod
    def preprocess_graph_new(adj, num_nodes):
        adj_norm = torch.eye(num_nodes, num_nodes)
        for row, col in adj.transpose(0, 1):
            adj_norm[row][col] = 1
        deg = torch.sum(adj_norm, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch.mul(adj_norm, deg_inv_sqrt.view(-1, 1))
        adj_t = torch.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t.to(adj.device)


# pfl
class PFL_Net(nn.Module):
    """ one layer of GCN """
    #emb_dim=8
    def __init__(self, in_channels, out_channels, hidden=64, emb_dim=8, n_hidden=1, bias=True,
                 ep=False):
        super(PFL_Net, self).__init__()
        self.input_dim = in_channels
        self.hidden_dim = hidden
        self.output_dim = out_channels
        self.embeddings = nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)
        layers = [
            nn.Linear(emb_dim, emb_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(emb_dim, emb_dim)
            )
        self.mlp = nn.Sequential(*layers)
        self.c1_weights = nn.Linear(emb_dim, self.input_dim * self.hidden_dim)  # input_dim*output_dim
        self.c1_bias = nn.Linear(emb_dim, self.hidden_dim)
        self.c2_weights = nn.Linear(emb_dim, self.hidden_dim * self.output_dim)  # input_dim*output_dim
        self.c2_bias = nn.Linear(emb_dim, self.output_dim)
        self.ep = ep
        self.reset_parameters()
        # self.adj = None

    def reset_parameters(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)
        # self.adj = None

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        weight = {
            "conv_a.W": self.c1_weights(features).view(self.input_dim, self.hidden_dim),
            "conv_a.b": self.c1_bias(features).view(-1),
            "conv_last.W": self.c2_weights(features).view(self.hidden_dim, self.output_dim),
            "conv_last.b": self.c2_bias(features).view(-1)}
        return weight


# 基础GCN
class GCNLayer(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.ep = ep
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj, ):  # model=1 表示正常模式，model=0表示生成图

        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        if self.ep:
            x = x @ x.T
        return x
