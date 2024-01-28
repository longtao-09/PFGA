import time
import torch
import torch.nn.functional as F

from misc.utils import *
from models.models_25 import *
from modules.federated import ClientModule


class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.rnd_local_val_acc, self.rnd_local_test_acc = None, None
        self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(g_id)
        self.parameters = list(self.model.parameters())

    def init_state(self):
        # self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        # self.optimizer = torch.optim.SGD(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        self.optimizer_rep = torch.optim.SGD(self.parameters, lr=1e-2, weight_decay=self.args.weight_decay)
        self.optimizer_cls = torch.optim.Adam(self.parameters, lr=self.args.base_lr,
                                              weight_decay=self.args.weight_decay)
        self.log = {
            'lr': [], 'train_lss': [],
            'ep_local_val_lss': [], 'ep_local_val_acc': [],
            'rnd_local_val_lss': [], 'rnd_local_val_acc': [],
            'ep_local_test_lss': [], 'ep_local_test_acc': [],
            'rnd_local_test_lss': [], 'rnd_local_test_acc': [],
        }
        self.model.reset_parameters()

    # def save_state(self):
    #     torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
    #         'optimizer': self.optimizer_cls.state_dict(),
    #         'model': get_state_dict(self.model),
    #         'log': self.log,
    #     })

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer_cls': self.optimizer_cls.state_dict(),
            'optimizer_rep': self.optimizer_rep.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        })

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']

    # def load_state1(self, loaded):
    #     set_state_dict(self.model, loaded['model'], self.gpu_id)
    #     self.optimizer.load_state_dict(loaded['optimizer'])
    #     self.log = loaded['log']
    #     self.model.adj = loaded['adj']
    #     self.model.adj_neg = loaded['adj_neg']
    #     self.model.adj_pos = loaded['adj_pos']
    #     self.model.graph_pos_edge, self.model.graph_neg_edge = loaded['graph_pos_edge'], loaded['graph_neg_edge']
    #     self.model.pos_adj_norm, self.model.neg_adj_norm = loaded['pos_adj_norm'], loaded['neg_adj_norm']
    #     self.model.pos_weight, self.model.neg_weight = loaded['pos_weight'], loaded['neg_weight']

    def load_state1(self, loaded):
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer_cls.load_state_dict(loaded['optimizer_cls'])
        self.optimizer_rep.load_state_dict(loaded['optimizer_rep'])
        self.log = loaded['log']
        self.model.adj = loaded['adj']
        self.model.adj_neg = loaded['adj_neg']
        self.model.adj_pos = loaded['adj_pos']
        self.model.graph_pos_edge, self.model.graph_neg_edge = loaded['graph_pos_edge'], loaded['graph_neg_edge']
        self.model.pos_adj_norm, self.model.neg_adj_norm = loaded['pos_adj_norm'], loaded['neg_adj_norm']
        self.model.pos_weight, self.model.neg_weight = loaded['pos_weight'], loaded['neg_weight']

    # def update_state(self, client_state, client_id):
    #     client_state[client_id] = {
    #         'optimizer': self.optimizer.state_dict(),
    #         'model': get_state_dict(self.model),
    #         'log': self.log,
    #         'adj': self.model.adj,
    #         'adj_neg': self.model.adj_neg,
    #         'adj_pos': self.model.adj_pos,
    #         'graph_pos_edge': self.model.graph_pos_edge,
    #         'graph_neg_edge': self.model.graph_neg_edge,
    #         'pos_adj_norm': self.model.pos_adj_norm,
    #         'neg_adj_norm': self.model.neg_adj_norm,
    #         'pos_weight': self.model.pos_weight,
    #         'neg_weight': self.model.neg_weight,
    #     }

    def update_state(self, client_state, client_id):
        client_state[client_id] = {
            'optimizer_cls': self.optimizer_cls.state_dict(),
            'optimizer_rep': self.optimizer_rep.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
            'adj': self.model.adj,
            'adj_neg': self.model.adj_neg,
            'adj_pos': self.model.adj_pos,
            'graph_pos_edge': self.model.graph_pos_edge,
            'graph_neg_edge': self.model.graph_neg_edge,
            'pos_adj_norm': self.model.pos_adj_norm,
            'neg_adj_norm': self.model.neg_adj_norm,
            'pos_weight': self.model.pos_weight,
            'neg_weight': self.model.neg_weight,
        }

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd['global'])

    def update(self, update):
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True)

    def on_round_begin(self, client_id):
        self.train()
        self.transfer_to_server()

    def loss_means(self, summary, label_all, num_classes):
        r"""Computes the margin objective."""
        std = 0
        for i in range(num_classes):
            node = summary[label_all == i]
            if node.shape[0] != 0:
                std += node.std()
        return std

    def train_rep(self, model, batch, num_classes, alpha=0.5, beta=3.0, gamma=2.0, train_edge=None, new_label=None):
        if isinstance(new_label, np.ndarray):
            label_all = new_label
        else:
            label_all = batch.y
        alpha = alpha  # MAX: Cora: 0.5,3,2,300,83.6,    0.58,  0.65,  3.4000,
        beta = beta  # 2
        gamma = gamma  # 2
        if train_edge is not None:
            adj = train_edge.train_pos_edge_index
            summary, summary_pos, summary_neg, loss_co = model.train_present(batch, label_all, adj)
        else:
            summary, summary_pos, summary_neg, loss_co = model.train_present(batch, label=label_all)
        loss_s = self.loss_means(summary, label_all, num_classes)
        # loss_cl = F.mse_loss(summary, summary_pos)
        loss_cl = torch.nn.functional.triplet_margin_loss(summary, summary_pos, summary_neg, reduction='mean')
        loss = beta * ((1 - alpha) * loss_cl + alpha * loss_co) + loss_s * gamma

        return loss

    def train(self):
        # cora 0.28,5,1.4
        alpha, beta, gamma = self.args.alpha, self.args.beta, self.args.gamma
        for ep in range(self.args.n_eps):
            st = time.time()
            self.model.train()
            # for _, batch in enumerate(self.loader.pa_loader):
            batch = self.loader.partition[0]
            # self.optimizer.zero_grad()
            self.optimizer_cls.zero_grad()
            self.optimizer_rep.zero_grad()
            from torch_geometric.utils import degree
            degree_count = degree(batch.edge_index[0], batch.num_nodes).cpu().detach().numpy()
            batch = batch.cuda(self.gpu_id)
            flag = np.median(degree_count)
            new_label = np.where(degree_count < flag, 0, 1)
            rep_loss = self.train_rep(self.model, batch, 2, alpha=alpha, beta=beta, gamma=gamma,
                                      new_label=new_label)
            rep_loss.backward()
            self.optimizer_rep.step()
            y_hat = self.model(batch)
            train_lss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
            train_lss.backward()
            self.optimizer_cls.step()
            # y_hat = self.model(batch)
            # train_lss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask]) + rep_loss
            # train_lss.backward()
            # self.optimizer.step()
            val_local_acc, val_local_lss = self.validate(mode='valid')
            test_local_acc, test_local_lss = self.validate(mode='test')
            # self.logger.print(
            #     f'rnd:{self.curr_rnd + 1}, ep:{ep + 1}, '
            #     f'val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f}, lr: {self.get_lr()} ({time.time() - st:.2f}s)'
            # )
            self.log['train_lss'].append(train_lss.item())
            self.log['ep_local_val_acc'].append(val_local_acc)
            self.log['ep_local_val_lss'].append(val_local_lss)
            self.log['ep_local_test_acc'].append(test_local_acc)
            self.log['ep_local_test_lss'].append(test_local_lss)
        self.log['rnd_local_val_acc'].append(val_local_acc)
        self.log['rnd_local_val_lss'].append(val_local_lss)
        self.log['rnd_local_test_acc'].append(test_local_acc)
        self.log['rnd_local_test_lss'].append(test_local_lss)
        self.rnd_local_val_acc = val_local_acc
        self.rnd_local_test_acc = test_local_acc
        self.save_log()

    def transfer_to_server(self):
        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'train_size': self.loader.train_size,
            'rnd_local_val_acc': self.rnd_local_val_acc,
            'rnd_local_test_acc': self.rnd_local_test_acc
        }
