import time
import numpy as np

from misc.utils import *
from models.models_25 import *
from modules.federated import ServerModule


class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        # TODO
        self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args, task=self.args.task).cuda(
            self.gpu_id)
        self.best_val_acc, self.last_test_acc, self.best_val_ap, self.last_test_ap = 0, 0, 0, 0

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()

    def calculate_average_test_accuracy(self, updated):
        total_test_acc = 0.0
        total_val_acc = 0.0
        total_test_ap = 0.0
        total_val_ap = 0.0
        num_clients = len(updated)
        for c_id in updated:
            client_val_acc, client_test_acc = self.sd[c_id]['rnd_local_val_acc'], self.sd[c_id]['rnd_local_test_acc']
            client_val_ap, client_test_ap = self.sd[c_id]['rnd_local_val_lss'], self.sd[c_id]['rnd_local_test_lss']
            total_test_acc += client_test_acc
            total_val_acc += client_val_acc
            total_test_ap += client_test_ap
            total_val_ap += client_val_ap
        average_val_acc = total_val_acc / num_clients
        average_test_acc = total_test_acc / num_clients
        average_test_ap = total_test_ap / num_clients
        average_val_ap = total_val_ap / num_clients
        if self.best_val_acc <= average_val_acc:
            self.best_val_acc = average_val_acc
            self.last_test_acc = average_test_acc
            self.best_val_ap = average_val_ap
            self.last_test_ap = average_test_ap

    def on_round_complete(self, updated):
        self.calculate_average_test_accuracy(updated)

        self.sd['acc'] = self.last_test_acc
        if self.args.print:
            if self.args.task:
                self.logger.print(
                    f'Avg val auc = {self.best_val_acc * 100:.1f}%, Avg val ap = {self.best_val_ap * 100:.1f}% '
                    f'Avg test auc = {self.last_test_acc * 100:.1f}%, Avg test ap = {self.last_test_ap * 100:.1f}%')
            else:
                self.logger.print(
                    f'Avg val acc:{self.best_val_acc * 100:.1f}%,Avg test acc: {self.last_test_acc * 100:.1f}%')
        # if self.curr_rnd + 1 in [1, 5, 10, 20, 50, 100]:
        #     self.logger.print(
        #         f'epoch:{self.curr_rnd + 1},Avg test acc: {self.last_test_acc * 100:.1f}%')
        if self.curr_rnd + 1 == self.args.n_rnds:
            if self.args.task:
                self.logger.print(
                    f'Avg val auc = {self.best_val_acc * 100:.1f}%, Avg val ap = {self.best_val_ap * 100:.1f}% '
                    f'Avg test auc = {self.last_test_acc * 100:.1f}%, Avg test ap = {self.last_test_ap * 100:.1f}%')
            else:
                self.logger.print(
                    f'Avg val acc:{self.best_val_acc * 100:.1f}%,Avg test acc: {self.last_test_acc * 100:.1f}%')
        # self.logger.print(f'Avg val acc:{self.best_val_acc * 100:.1f}%,Avg test acc: {self.last_test_acc * 100:.1f}%')
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
        # self.logger.print(f'all clients have been uploaded ({time.time() - st:.2f}s)')
        st = time.time()
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        # self.logger.print(f'global model has been updated ({time.time() - st:.2f}s)')
        if self.args.print:
            self.logger.print(f'all clients have been uploaded ({time.time() - st:.2f}s)')

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
