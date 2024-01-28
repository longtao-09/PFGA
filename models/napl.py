import torch
import torch.nn.functional as F
from torch import nn


class GCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=64, n_clss=10, args=None, dropout=0.5, emb=32):
        super(GCN, self).__init__()
        # GCNLayer_0:节点个性化, _1:客户端个性化，_2:基础GCN， GCNConv：库卷积 _3:不用输入产生参数n*d _4：不用输入产生参数1*d
        if args.acg_model == 0:
            self.conv_a = GCNLayer_2(input_dim=n_feat, output_dim=n_dims, activation=F.relu, dropout=dropout)
        elif args.acg_model == 1:
            self.conv_a = GCNLayer_0(input_dim=n_feat, output_dim=n_dims, activation=F.relu, dropout=dropout)
        elif args.acg_model == 2:
            self.conv_a = GCNLayer_1(input_dim=n_feat, output_dim=n_dims, activation=F.relu, dropout=dropout)
        self.conv_last = GCNLayer_2(input_dim=n_dims, output_dim=n_clss, activation=0, dropout=0)
        self.dropout = dropout
        self.adj = None

    def reset_parameters(self):
        self.conv_a.init_params()
        self.conv_last.init_params()
        self.adj = None

    def forward(self, data):
        from torch_geometric.data import Data
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        if self.adj is None:
            self.adj = self.preprocess_graph_new(edge_index, x.size(0))
        # print('!!!!!!!!!!!!!!!x and adj', x.size(0), self.adj.shape)
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
        # return adj_t


# 不使用输入产生参数 1 d d c f
class GCNLayer_4(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer_4, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = 8
        self.W1 = nn.Parameter(torch.FloatTensor(output_dim // self.c))  # d
        self.W2 = nn.Parameter(torch.FloatTensor(output_dim // self.c, input_dim, output_dim))  # d*c*f
        self.activation = activation
        self.ep = ep
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim // self.c, output_dim))
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
        # node = h @ self.W1
        # node = node.mean(dim=0)
        node = self.W1
        # w = node @ self.W2
        w = torch.einsum('d,dcf->cf', node, self.W2)
        b = torch.matmul(node, self.b)
        # x = x @ self.W2
        x = h @ w
        x = adj @ x
        if self.b is not None:
            x = x + b
        if self.activation:
            x = self.activation(x)
        if self.ep:
            x = x @ x.T
        return x


# 不使用输入产生参数 n d d c f
class GCNLayer_3(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer_3, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = 8
        self.W1 = None
        self.W2 = nn.Parameter(torch.FloatTensor(output_dim // self.c, input_dim, output_dim))  # d*c*f
        self.activation = activation
        self.ep = ep
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim // self.c, output_dim))
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
        self.W1 = None

    def forward(self, h, adj, ):  # model=1 表示正常模式，model=0表示生成图
        if self.dropout:
            h = self.dropout(h)
        if self.W1 is None:
            self.W1 = nn.Parameter(torch.FloatTensor(h.shape[0], self.output_dim // self.c)).to(h.device)  # n*d
        node = self.W1
        # node = h @ self.W1
        # node = node.mean(dim=0)
        # w = node @ self.W2
        w = torch.einsum('nd,dcf->ncf', node, self.W2)
        b = torch.matmul(node, self.b)
        x = torch.einsum('nc,ncf->nf', h, w)
        # x = x @ self.W2
        # x = h @ w
        x = adj @ x
        if self.b is not None:
            x = x + b
        if self.activation:
            x = self.activation(x)
        if self.ep:
            x = x @ x.T
        return x


# 节点个性化
class GCNLayer_0(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer_0, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = 8
        self.W1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim // self.c))  # c*d
        self.W2 = nn.Parameter(torch.FloatTensor(output_dim // self.c, input_dim, output_dim))  # d*c
        self.activation = activation
        self.ep = ep
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim // self.c, output_dim))
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
        # nc cd =nd , nd dcf = ncf,
        node = h @ self.W1
        # node = adj @ node
        # node = node.mean(dim=0)
        # w = node @ self.W2
        w = torch.einsum('nd,dcf->ncf', node, self.W2)
        b = torch.matmul(node, self.b)
        x = torch.einsum('nc,ncf->nf', h, w)
        # x = x @ self.W2
        # x = h @ w
        x = adj @ x
        if self.b is not None:
            x = x + b
        if self.activation:
            x = self.activation(x)
        if self.ep:
            x = x @ x.T
        return x


# 客户端个性化
class GCNLayer_1(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer_1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = 8
        self.W1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim // self.c))  # c*d
        self.W2 = nn.Parameter(torch.FloatTensor(output_dim // self.c, input_dim, output_dim))  # d*c*f
        self.activation = activation
        self.ep = ep
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim // self.c, output_dim))
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
        node = h @ self.W1
        # node = adj @ node
        node = node.mean(dim=0)
        # w = node @ self.W2
        w = torch.einsum('d,dcf->cf', node, self.W2)
        b = torch.matmul(node, self.b)
        # x = x @ self.W2
        x = h @ w
        x = adj @ x
        if self.b is not None:
            x = x + b
        if self.activation:
            x = self.activation(x)
        if self.ep:
            x = x @ x.T
        return x


# 基础GCN
class GCNLayer_2(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer_2, self).__init__()
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

    def forward(self, h, adj):  # model=1 表示正常模式，model=0表示生成图

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


# 节点个性化
class GCNLayer_0(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer_0, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = 8
        self.W1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim // self.c))  # c*d
        self.W2 = nn.Parameter(torch.FloatTensor(output_dim // self.c, input_dim, output_dim))  # d*c
        self.activation = activation
        self.ep = ep
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim // self.c, output_dim))
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

    def forward(self, h, adj, model=0):  # model=1 表示正常模式，model=0表示生成图
        if self.dropout:
            h = self.dropout(h)
        node = h @ self.W1
        # node = node.mean(dim=0)
        # w = node @ self.W2
        w = torch.einsum('nd,dcf->ncf', node, self.W2)
        x = torch.einsum('nc,ncf->nf', h, w)
        # x = x @ self.W2
        # x = h @ w
        x = adj @ x
        if self.b is not None:
            b = torch.matmul(node, self.b)
            x = x + b
        if self.activation:
            x = self.activation(x)
        if self.ep:
            x = x @ x.T
        return x


# 客户端个性化
class GCNLayer_1(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer_1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = 8
        self.W1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim // self.c))  # c*d
        self.W2 = nn.Parameter(torch.FloatTensor(output_dim // self.c, input_dim, output_dim))  # d*c
        self.activation = activation
        self.ep = ep
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim // self.c, output_dim))
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

    def forward(self, h, adj, model=0):  # model=1 表示正常模式，model=0表示生成图
        if self.dropout:
            h = self.dropout(h)
        node = h @ self.W1
        node = node.mean(dim=0)
        # w = node @ self.W2
        w = torch.einsum('d,dcf->cf', node, self.W2)
        # x = x @ self.W2
        x = h @ w
        x = adj @ x
        if self.b is not None:
            b = torch.matmul(node, self.b)
            x = x + b
        if self.activation:
            x = self.activation(x)
        if self.ep:
            x = x @ x.T
        return x