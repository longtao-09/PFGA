import copy
import os
import glob
import json
import random
import numpy as np

from collections import defaultdict, OrderedDict
from misc.forked_pdb import ForkedPdb
from torch_geometric.utils import degree, train_test_split_edges

import torch
from torch import Tensor


def str2bool(v):
    return v.lower() in ['true', 't']


def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)
    torch.save(data, fpath)


def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)
    return torch.load(fpath, map_location=torch.device('cpu'))


def shuffle(seed, x, y):
    idx = np.arange(len(x))
    random.seed(seed)
    random.shuffle(idx)
    return [x[i] for i in idx], [y[i] for i in idx]


def save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, filename), 'w+') as outfile:
        json.dump(data, outfile)


def exists(base_dir, filename):
    return os.path.exists(os.path.join(base_dir, filename))


def join_glob(base_dir, filename):
    return glob.glob(os.path.join(base_dir, filename))


def remove_if_exist(base_dir, filename):
    targets = join_glob(base_dir, filename)
    if len(targets) > 0:
        for t in targets:
            os.remove(t)


def debugger():
    ForkedPdb().set_trace()


def get_state_dict(model):
    state_dict = convert_tensor_to_np(model.state_dict())
    return state_dict


def set_state_dict(model, state_dict, gpu_id, skip_stat=False, skip_mask=False, Local=False):
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip_stat=skip_stat, skip_mask=skip_mask,
                                      model=model.state_dict())
    # state_mask = ['conv_a.W1', 'conv_a.W2', 'conv_a.b']

    if Local:
        state_mask = ['cons.W', 'cons.b', 'cls.W', 'cls.b']
        # state_mask = ['cons.W1', 'cons.W2', 'cons.b', 'cls.W', 'cls.b']
        # # state_mask = ['conv_last.W', 'conv_last.b']
        for mask in state_mask:
            state_dict[mask] = model.state_dict()[mask]
    # if Local:
    #     all_mask = [k for k, _ in state_dict.items()]
    #     state_mask = ['cons.W', 'cons.b', 'cls.W', 'cls.b']
    #     # state_mask = ['cons.W1', 'cons.W2', 'cons.b', 'cls.W', 'cls.b']
    #     # # state_mask = ['conv_last.W', 'conv_last.b']
    #     for mask in all_mask:
    #         if mask not in state_mask:
    #             state_dict[mask] = model.state_dict()[mask]
    model.load_state_dict(state_dict)


def convert_tensor_to_np(state_dict):
    return OrderedDict([(k, v.clone().detach().cpu().numpy()) for k, v in state_dict.items()])


def convert_np_to_tensor(state_dict, gpu_id, skip_stat=False, skip_mask=False, model=None):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        if skip_stat:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue
        if skip_mask:
            if 'mask' in k or 'pre' in k or 'pos' in k:
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = torch.tensor(v).cuda(gpu_id)
        else:
            _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id)
    return _state_dict


def convert_np_to_tensor_cpu(state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        _state_dict[k] = torch.tensor(v)
    return _state_dict


def get_ep_data(data, args):
    # new_label = None
    # a = data.edge_index
    if args.task:
        train_rate = 0.85
        val_ratio = (1 - train_rate) / 3
        test_ratio = (1 - train_rate) / 3 * 2
        train_edge = train_test_split_edges(copy.copy(data), val_ratio, test_ratio)
        adj = train_edge.train_pos_edge_index.cuda()
        adj_m = np.zeros((data.x.size(0), data.x.size(0)))
        for i in range(adj.size(1)):
            adj_m[adj[0][i]][adj[1][i]] = 1
            adj_m[adj[1][i]][adj[0][i]] = 1
        adj_m = torch.from_numpy(adj_m)
        # pos_weight, norm_w = compute_loss_para(adj_m)
        norm_w = adj_m.shape[0] ** 2 / float((adj_m.shape[0] ** 2 - adj_m.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_m.shape[0] ** 2 - adj_m.sum()) / adj_m.sum()])
        new_adj = adj_m.to_sparse().indices()
        # if args.task == 1:
        degree_count = degree(new_adj[0]).numpy()
        flag = np.median(degree_count)
        # 将 degree_count 中与 flag 的关系进行值替换的位置替换为随机数组中的对应值
        # new_label = np.where(degree_count == flag, random_values, degree_count)
        # new_label = np.where(degree_count == flag, np.random.choice([0, 1], size=len(degree_count)),
        #                      np.where(degree_count < flag, 0, 1))
        new_label = np.where(degree_count < flag, 0, 1)
        return new_label, adj_m, norm_w, pos_weight, train_edge
    else:
        degree_count = degree(data.edge_index[0], data.num_nodes).numpy()
        # unique_elements, counts = np.unique(degree_count, return_counts=True)
        # print(f'avg degree:{np.mean(unique_elements)}')
        # print(f'unique_elements:{unique_elements},counts:{counts}')
        flag = np.median(degree_count)
        # 将 degree_count 中与 flag 的关系进行值替换的位置替换为随机数组中的对应值
        # new_label = np.where(degree_count == flag, random_values, degree_count)
        # new_label = np.where(degree_count == flag, np.random.choice([0, 1], size=len(degree_count)),
        #                      np.where(degree_count < flag, 0, 1))
        new_label = np.where(degree_count < flag, 0, 1)
        return new_label


def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    import networkx as nx
    from torch_geometric.data import Data

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data
