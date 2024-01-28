import os

import torch
import random
import numpy as np

import metispy as metis

import torch_geometric
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Coauthor, Planetoid, Amazon
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils import get_data, split_train, torch_save

data_path = '../../datasets'
ratio_train = 0.2
seed = 1234
# comms = [2, 6, 10]
comms = [2, 6, 10]
# n_clien_per_comm = 2
n_clien_per_comm = 5
data_all = ['Cora', 'Citeseer', 'Pubmed', 'coauthor-cs', 'amazon-computers', 'amazon-photo', 'ogbn-arxiv']
INF = np.iinfo(np.int64).max

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def generate_data(dataset, n_comms):
    os.makedirs(f'../../datasets/{dataset}_overlapping/10/', exist_ok=True)
    os.makedirs(f'../../datasets/{dataset}_overlapping/30/', exist_ok=True)
    os.makedirs(f'../../datasets/{dataset}_overlapping/50/', exist_ok=True)
    # data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'overlapping',
    #                    n_comms * n_clien_per_comm)
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
        # count = int(data.x.size(0) * 0.8)
        # node = torch.randint(0, data.x.size(0), (count,))
        # data = data.subgraph(node)

        num_classes = torch.unique(data.y).size(0)
        data.y = data.y.squeeze()
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
    split_subgraphs(n_comms, data, dataset)


def split_subgraphs(n_comms, data, dataset):
    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_comms)
    assert len(list(set(membership))) == n_comms
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')

    adj = to_dense_adj(data.edge_index)[0]
    for comm_id in range(n_comms):
        for client_id in range(n_clien_per_comm):
            client_indices = np.where(np.array(membership) == comm_id)[0]
            client_indices = list(client_indices)
            client_num_nodes = len(client_indices)

            # client_indices = random.sample(client_indices, client_num_nodes // 2)

            # ogbn-arxiv:
            client_indices = random.sample(client_indices, client_num_nodes // (n_clien_per_comm - 1))
            client_num_nodes = len(client_indices)

            client_edge_index = []
            client_adj = adj[client_indices][:, client_indices]
            client_edge_index, _ = dense_to_sparse(client_adj)
            client_edge_index = client_edge_index.T.tolist()
            client_num_edges = len(client_edge_index)

            client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
            client_x = data.x[client_indices]
            client_y = data.y[client_indices]
            # print("---------", client_x.shape, client_y.shape, "---------")
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

            torch_save(data_path,
                       f'{dataset}_overlapping/{n_comms * n_clien_per_comm}/partition_{comm_id * n_clien_per_comm + client_id}.pt',
                       {
                           'client_data': client_data,
                           'client_id': client_id
                       })
            print(
                f'client_id: {comm_id * n_clien_per_comm + client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')


for dataset in data_all[-1:]:
    for n_comms in comms:
        print(f'dataset:{dataset},n_client:{n_clien_per_comm * n_comms}')
        generate_data(dataset=dataset, n_comms=n_comms)
