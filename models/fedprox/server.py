import time
import numpy as np

from misc.utils import *
from models.nets import *
from modules.federated import ServerModule


class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.gpu_id)
        self.best_val_acc, self.last_test_acc = 0, 0
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.base_lr,
        #                                   weight_decay=self.args.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.base_lr,
                                          weight_decay=self.args.weight_decay)

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()
        self.sd['global_models'] = self.model

    def calculate_average_test_accuracy(self, updated):
        total_test_acc = 0.0
        total_val_acc = 0.0
        num_clients = len(updated)
        for c_id in updated:
            client_val_acc, client_test_acc = self.sd[c_id]['rnd_local_val_acc'], self.sd[c_id]['rnd_local_test_acc']
            total_test_acc += client_test_acc
            total_val_acc += client_val_acc
        average_val_acc = total_val_acc / num_clients
        average_test_acc = total_test_acc / num_clients
        if self.best_val_acc <= average_val_acc:
            self.best_val_acc = average_val_acc
            self.last_test_acc = average_test_acc

    def on_round_complete(self, updated):
        self.calculate_average_test_accuracy(updated)
        if self.args.print:
            self.logger.print(f'Avg val acc:{self.best_val_acc * 100:.1f}%,Avg test acc: {self.last_test_acc * 100:.1f}%')
        if self.curr_rnd+1 == self.args.n_rnds:
            self.logger.print(
                f'Avg val acc:{self.best_val_acc * 100:.1f}%,Avg test acc: {self.last_test_acc * 100:.1f}%')
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()
        local_weights = []
        local_train_sizes = []
        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        if self.args.print:
            self.logger.print(f'all clients have been uploaded ({time.time() - st:.2f}s)')
        st = time.time()
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        if self.args.print:
            self.logger.print(f'global model has been updated ({time.time() - st:.2f}s)')

    def compute_local_loss(self, client_model, global_model_params):
        local_loss = 0.0
        for client_param, global_param in zip(client_model, global_model_params):
            # 计算参数之间的L2范数差异
            param_diff = client_param - global_param
            local_loss += torch.norm(param_diff, p=2)  # 这里使用L2范数
        return local_loss

    def update_new(self, updated):
        st = time.time()
        local_weights = []
        local_train_sizes = []
        global_loss = 0
        proximal_term = 0.1
        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_train_sizes.append(self.sd[c_id]['train_size'])
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        for c_id in updated:
            global_loss += self.compute_local_loss(self.sd[c_id]['models'].parameters(), self.model.parameters())
            # 添加近端项
            proximal_term_loss = 0.0
            for param1, param2 in zip(self.model.parameters(), self.sd[c_id]['models'].parameters()):
                proximal_term_loss += torch.norm(param1 - param2)
            global_loss += proximal_term * proximal_term_loss
            # del self.sd[c_id]
        if self.args.print:
            self.logger.print(f'all clients have been uploaded ({time.time() - st:.2f}s)')

        self.optimizer.zero_grad()
        global_loss.backward()
        self.optimizer.step()
        st = time.time()
        if self.args.print:
            self.logger.print(f'global model has been updated ({time.time() - st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model)
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
        })
