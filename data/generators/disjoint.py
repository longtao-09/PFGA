import torch
import random
import numpy as np
import os
import metispy as metis

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor, WikiCS, Amazon
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Reddit
from misc.plot import plot_visual

from utils import get_data, split_train, torch_save

data_path = '../../datasets'
ratio_train = 0.2
seed = 1234
clients = [5, 10, 20]
data_all = ['Cora', 'Citeseer', 'Pubmed', 'coauthor-cs', 'amazon-computers', 'amazon-photo', 'ogbn-arxiv']

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
INF = np.iinfo(np.int64).max


# train,val,test: 6,2,2
def generate_data(dataset, n_clients):
    os.makedirs(f'../../datasets/{dataset}_all/', exist_ok=True)
    if dataset in ['Cora', 'Citeseer', 'Pubmed']:
        num_split = {
            'Cora': [232, 542, INF],
            'Citeseer': [332, 665, INF],
            'Pubmed': [3943, 3943, INF],
        }
        data = Planetoid(data_path,
                         dataset,
                         split='random',
                         num_train_per_class=num_split[dataset][0],
                         num_val=num_split[dataset][1],
                         num_test=num_split[dataset][2])[0]
        # data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'disjoint', n_clients)
        # split_subgraphs(n_clients, data, dataset)

    if dataset == 'coauthor-cs':
        data = Coauthor(root=data_path + dataset, name='cs')  # , transform=T.NormalizeFeatures())
        data = data[0]
        num_classes = torch.unique(data.y).size(0)
        features = data.x
        print(f'{dataset}:features:{features.size()},num classes:{num_classes}')
        num_nodes = features.size(0)
        flag_train = int(num_nodes * 0.6)
        flag_val = int(num_nodes * 0.8)
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[:flag_train] = True
        val_mask[flag_train:flag_val] = True
        test_mask[flag_val:] = True
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(
            test_mask)

    elif dataset == 'amazon-computers':
        data = Amazon(root=data_path, name='computers')  # , transform=T.NormalizeFeatures())
        data = data[0]
        num_classes = torch.unique(data.y).size(0)
        features = data.x
        print(f'{dataset}:features:{features.size()},num classes:{num_classes}')
        features = data.x
        num_nodes = features.size(0)
        flag_train = int(num_nodes * 0.6)
        flag_val = int(num_nodes * 0.8)
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[:flag_train] = True
        val_mask[flag_train:flag_val] = True
        test_mask[flag_val:] = True
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(
            test_mask)

    elif dataset == 'amazon-photo':
        data = Amazon(root=data_path, name='photo')  # , transform=T.NormalizeFeatures())
        data = data[0]
        num_classes = torch.unique(data.y).size(0)
        features = data.x
        print(f'{dataset}:features:{features.size()},num classes:{num_classes}')
        features = data.x
        num_nodes = features.size(0)
        flag_train = int(num_nodes * 0.6)
        flag_val = int(num_nodes * 0.8)
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[:flag_train] = True
        val_mask[flag_train:flag_val] = True
        test_mask[flag_val:] = True
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(
            test_mask)

    elif dataset.startswith('ogbn'):
        data = PygNodePropPredDataset(root=data_path + dataset, name=dataset)
        data = data[0]
        count = int(data.x.size(0) * 0.9)
        node = torch.randint(0, data.x.size(0), (count,))
        data = data.subgraph(node)

        num_classes = torch.unique(data.y).size(0)
        features = data.x
        print(f'{dataset}:features:{features.size()},num classes:{num_classes}')

        num_nodes = features.size(0)
        # flag_train = int(num_nodes * 0.6)
        # flag_val = int(num_nodes * 0.8)
        flag_train = int(num_nodes * 0.1)
        flag_val = int(num_nodes * 0.55)
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[:flag_train] = True
        val_mask[flag_train:flag_val] = True
        test_mask[flag_val:] = True
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(
            test_mask)
    elif dataset == 'reddit':
        # 加载Reddit社交图数据集
        data = Reddit(root=data_path + dataset)
        data = data[0]

        count = int(data.x.size(0) * 0.6)
        node = torch.randint(0, data.x.size(0), (count,))
        data = data.subgraph(node)

        num_classes = torch.unique(data.y).size(0)
        features = data.x
        print(f'{dataset}:features:{features.size()},num classes:{num_classes}')

        num_nodes = features.size(0)
        # flag_train = int(num_nodes * 0.6)
        # flag_val = int(num_nodes * 0.8)
        flag_train = int(num_nodes * 0.1)
        flag_val = int(num_nodes * 0.55)
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[:flag_train] = True
        val_mask[flag_train:flag_val] = True
        test_mask[flag_val:] = True
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(
            test_mask)

    torch_save(data_path, f'{dataset}_all/{dataset}_all.pt', {
        'data': data,
    })
    split_subgraphs_new(n_clients, data, dataset)
    # split_subgraphs_class_imbalance(n_clients, data, dataset)


def split_subgraphs_new(n_clients, data, dataset, ovlap=0):
    os.makedirs(f'../../datasets/{dataset}_disjoint/{n_clients}/', exist_ok=True)
    sampling_rate = (np.ones(n_clients) - ovlap) / n_clients
    data.index_orig = torch.arange(data.num_nodes)
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")

    client_node_idx = {idx: [] for idx in range(n_clients)}

    indices = np.random.permutation(data.num_nodes)
    sum_rate = 0
    for idx, rate in enumerate(sampling_rate):
        client_node_idx[idx] = indices[round(sum_rate *
                                             data.num_nodes):round(
            (sum_rate + rate) *
            data.num_nodes)]
        sum_rate += rate
    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        graphs.append(from_networkx(sub_g))
    for client_id in range(n_clients):
        torch_save(data_path, f'{dataset}_disjoint/{n_clients}/partition_{client_id}.pt', {
            'client_data': graphs[client_id],
            'client_id': client_id
        })
        print(
            f'client_id: {client_id}, iid, n_train_node: {graphs[client_id].x.shape[0]}, n_train_edge: {graphs[client_id].edge_index.shape[1]}')
    del_edge(data, client_node_idx, n_clients, dataset)


def split_subgraphs_class_imbalance(n_clients, data, dataset, imbalance_ratio=0.7):
    os.makedirs(f'../../datasets/{dataset}_imbalance/{n_clients}/', exist_ok=True)
    y = data.y
    class_counts = torch.bincount(y)
    n_classes = len(class_counts)

    data_list_res = [[] for _ in range(n_clients)]

    # 将每一类分成两部分，一部分0.7，一部分0.3，分别组合起来
    for label in range(n_classes):
        class_samples = (y == label).nonzero().flatten().tolist()
        random.shuffle(class_samples)

        samples_07 = int(len(class_samples) * imbalance_ratio)
        samples_03 = len(class_samples) - samples_07

        data_list_res_temp = []
        data_list_res_temp.extend(class_samples[:samples_07])
        random.shuffle(class_samples[samples_07:])
        data_list_res_temp.extend(class_samples[samples_07:samples_07 + samples_03])

        # 平均分配给每个客户端
        for i, sample in enumerate(data_list_res_temp):
            client_id = i % n_clients
            data_list_res[client_id].append(sample)
    # 3, 5:0.6，0.3，0.3，0.3
    data.index_orig = torch.arange(data.num_nodes)
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")

    client_node_idx = {idx: [] for idx in range(n_clients)}

    for idx in range(n_clients):
        client_node_idx[idx] = data_list_res[idx]

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        graphs.append(from_networkx(sub_g))

    for client_id in range(n_clients):
        torch_save(data_path, f'{dataset}_imbalance/{n_clients}/partition_{client_id}.pt', {
            'client_data': graphs[client_id],
            'client_id': client_id
        })
        print(
            f'client_id: {client_id}, n_train_node: {graphs[client_id].x.shape[0]}, n_train_edge: {graphs[client_id].edge_index.shape[1]}')

    # del_edge(data, client_node_idx, n_clients, dataset)


def split_subgraphs_noiid(n_clients, data, dataset, ovlap=0):
    y = data.y
    data_list_res = [[] for i in range(n_clients)]
    for i, label in enumerate(y):
        if random.random() <= 0.8:
            data_list_res[label].append(i)
        else:
            flag = random.randint(0, n_clients - 1)
            if flag >= label:
                flag += 1
            data_list_res[flag].append(i)

    data.index_orig = torch.arange(data.num_nodes)
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")

    client_node_idx = {idx: [] for idx in range(n_clients)}

    for idx in range(n_clients):
        client_node_idx[idx] = data_list_res[idx]

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        graphs.append(from_networkx(sub_g))
    for client_id in range(n_clients):
        torch_save(data_path, f'{dataset}_imbalance/{n_clients}/partition_{client_id}.pt', {
            'client_data': graphs[client_id],
            'client_id': client_id
        })
        print(
            f'client_id: {client_id}, iid, n_train_node: {graphs[client_id].x.shape[0]}, n_train_edge: {graphs[client_id].edge_index.shape[1]}')
    # del_edge(data, client_node_idx, n_clients, dataset)


def del_edge(data, client_node_idx, n_clients, dataset):
    edge = data.edge_index
    edge = edge.transpose(1, 0)
    del_columns = []
    for ind, (i, j) in enumerate(edge):
        flag = 0
        while int(i) not in client_node_idx[flag]:
            flag += 1
        if int(j) not in client_node_idx[flag]:
            del_columns.append(ind)
    edge = edge.transpose(1, 0)
    selected_columns = [i for i in range(edge.size(1)) if i not in del_columns]
    data.edge_index = edge[:, selected_columns]
    torch_save(data_path, f'{dataset}_disjoint/{n_clients}/drop_edge.pt', {
        'data': data,
    })


def split_subgraphs(n_clients, data, dataset):
    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_clients)
    assert len(list(set(membership))) == n_clients
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')

    adj = to_dense_adj(data.edge_index)[0]
    for client_id in range(n_clients):
        client_indices = np.where(np.array(membership) == client_id)[0]
        client_indices = list(client_indices)
        client_num_nodes = len(client_indices)

        client_edge_index = []
        client_adj = adj[client_indices][:, client_indices]
        client_edge_index, _ = dense_to_sparse(client_adj)
        client_edge_index = client_edge_index.T.tolist()
        client_num_edges = len(client_edge_index)

        client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
        client_x = data.x[client_indices]
        client_y = data.y[client_indices]
        client_train_mask = data.train_mask[client_indices]
        client_val_mask = data.val_mask[client_indices]
        client_test_mask = data.test_mask[client_indices]

        client_data = Data(
            x=client_x,
            y=client_y,
            edge_index=client_edge_index.t().contiguous(),
            train_mask=client_train_mask,
            val_mask=client_val_mask,
            test_mask=client_test_mask
        )
        assert torch.sum(client_train_mask).item() > 0

        torch_save(data_path, f'{dataset}_disjoint/{n_clients}/partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id
        })
        print(f'client_id: {client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')


for data_name in data_all[:3]:
    for n_clients in clients:
        generate_data(dataset=data_name, n_clients=n_clients)
