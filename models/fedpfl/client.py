import time
import torch
import torch.nn.functional as F

from misc.utils import *
from models.pfl import *
from modules.federated import ClientModule


class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.rnd_local_val_acc, self.rnd_local_test_acc = None, None
        self.model = PFL_Net(self.args.n_feat, self.args.n_clss, self.args.n_dims).cuda(g_id)
        self.parameters = list(self.model.parameters())

    def init_state(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        # self.optimizer = torch.optim.SGD(self.parameters, lr=self.args.base_lr, momentum=0.9, weight_decay=self.args.weight_decay)
        self.log = {
            'lr': [], 'train_lss': [],
            'ep_local_val_lss': [], 'ep_local_val_acc': [],
            'rnd_local_val_lss': [], 'rnd_local_val_acc': [],
            'ep_local_test_lss': [], 'ep_local_test_acc': [],
            'rnd_local_test_lss': [], 'rnd_local_test_acc': [],
        }
        self.model.reset_parameters()

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        })

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']

    def load_state1(self, loaded):
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']
        # self.model.adj = loaded['adj']

    def update_state(self, client_state, client_id):
        client_state[client_id] = {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
            # 'adj': self.model.adj,
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

    def train(self):
        for ep in range(self.args.n_eps):
            st = time.time()
            self.model.train()
            # for _, batch in enumerate(self.loader.pa_loader):
            batch = self.loader.partition[0]
            self.optimizer.zero_grad()
            batch = batch.cuda(self.gpu_id)
            weights = self.model(torch.tensor([0], dtype=torch.long).cuda(self.gpu_id))
            net = GCN_Net(batch.x.size(1), self.args.n_clss, self.args.n_dims).cuda(self.gpu_id)
            net.load_state_dict(weights)
            # init inner optimizer

            # storing theta_i for later calculating delta theta
            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})
            inner_steps = 50
            # inner_steps = 20
            inner_optim = torch.optim.SGD(
                net.parameters(), lr=1e-2, momentum=.9, weight_decay=5e-4)
            # inner_optim = torch.optim.Adam(
            #     net.parameters(), lr=1e-2,  weight_decay=5e-4)
            train_lss = torch.tensor(0.0, requires_grad=True).cuda(self.gpu_id)
            for i in range(inner_steps):
                net.train()
                inner_optim.zero_grad()
                y_hat = net(batch)
                train_lss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
                train_lss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
                inner_optim.step()
            net.eval()
            final_state = net.state_dict()
            self.optimizer.zero_grad()
            # calculating delta theta
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            # calculating phi gradient
            hnet_grads = torch.autograd.grad(
                list(weights.values()), self.model.parameters(), grad_outputs=list(delta_theta.values())
            )
            # update hnet weights
            for p, g in zip(self.model.parameters(), hnet_grads):
                p.grad = g

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
            self.optimizer.step()
            val_local_acc, val_local_lss = self.validate(mode='valid', model=net)
            test_local_acc, test_local_lss = self.validate(mode='test', model=net)
            if self.args.print:
                self.logger.print(
                    f'rnd:{self.curr_rnd + 1}, ep:{ep + 1}, '
                    f'val_local_loss: {val_local_lss.item():.4f}, '
                    f'val_local_acc: {val_local_acc:.4f}, lr: {self.get_lr()} ({(time.time() - st) * 1000:.4f}ms)'
                )
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
