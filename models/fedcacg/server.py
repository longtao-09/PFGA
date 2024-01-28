import time
import warnings
from datetime import datetime
import numpy as np
import torch
from torch import cosine_similarity

from misc.utils import *
from models.models_25 import *
from modules.federated import ServerModule
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.gpu_id)
        # self.model1 = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.gpu_id)
        self.best_val_acc, self.last_test_acc = 0, 0
        if args.n_clients == 5:
            self.center = 2
        elif args.n_clients == 10:
            self.center = 4
        elif args.n_clients == 20:
            self.center = 8
        elif args.n_clients == 30:
            self.center = 12
        elif args.n_clients == 50:
            self.center = 20
        self.cluster_labels = np.random.randint(self.center, size=args.n_clients)  # init
        self.model_parameters = [get_state_dict(self.model)] * (self.center + 1)

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()

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
            self.logger.print(
                f'Avg val acc:{self.best_val_acc * 100:.1f}%,Avg test acc: {self.last_test_acc * 100:.1f}%')
        if self.curr_rnd + 1 == self.args.n_rnds:
            self.logger.print(
                f'Avg val acc:{self.best_val_acc * 100:.1f}%,Avg test acc: {self.last_test_acc * 100:.1f}%')
        # self.logger.print(f'Avg val acc:{self.best_val_acc * 100:.1f}%,Avg test acc: {self.last_test_acc * 100:.1f}%')
        self.update(updated)
        # self.save_state()

    def km_cluseter(self, local_parameters):
        # 创建K均值聚类模型，设置簇数为2,全部参数都写上了，可以通过GPU甲酸，减少最大迭代轮数的方式进行加速
        # start_time = datetime.now()
        state_mask = ['conv_pos.gcn_base.W1', 'conv_pos.gcn_base.W2', 'conv_pos.gcn_mean.W', 'conv_pos.gcn_logist.W']

        params_info = OrderedDict([(k, None) for k in local_parameters[0].keys()])
        model_parameters = []
        for model_client in local_parameters:
            temp_parameters = []
            for name, params in params_info.items():
                if name in state_mask:
                    temp_parameters.extend(model_client[name].flatten())
            model_parameters.append(temp_parameters)
        # st = time.time()
        # 将模型参数转换为GPU上的tensor
        models_gpu = [torch.Tensor(model).view(1, -1).cuda(self.gpu_id) for model in model_parameters]
        # 计算模型中心
        center = torch.stack(models_gpu).mean(dim=0).view(1, -1).cuda(self.gpu_id)
        # 计算每个模型到中心的余弦相似度
        # similarities = [F.cosine_similarity(model, center).cpu().item() for model in models_gpu]
        similarities = [F.cosine_similarity(model, center).cpu().view(1, -1).item() for model in models_gpu]
        # print(f'-------------------------{similarities}----------------------')
        # threshold = np.median(similarities)
        # self.cluster_labels = np.where(similarities < threshold, 0, 1)
        similarities = np.array(similarities).reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.center, max_iter=20)
        kmeans.fit(similarities)

        # 获取每个模型的所属簇标签
        self.cluster_labels = kmeans.labels_
        # self.logger.print(f'clients cluster have been done!!!! ({time.time() - st:.2f}s)')
        # kmeans = KMeans_fun(k=2, max_iters=100)
        # # 获取每个模型的所属簇标签
        # _, self.cluster_labels = kmeans(model_parameters)
        # end_time = datetime.now()
        # print(f'Kmeans:{end_time - start_time}!!!!!!!!!!!!!!!!!!!!')

    def update(self, updated):
        local_weights = []
        local_train_sizes = []
        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        # self.logger.print(f'all clients have been uploaded ({time.time() - st:.2f}s)')
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.model_parameters[0] = self.aggregate(local_weights, ratio)
        self.km_cluseter(local_weights)
        # st = time.time()
        for label in range(max(self.cluster_labels)):
            local_train_sizes_temp = [local_train_sizes[i] for i in range(len(local_train_sizes)) if
                                      self.cluster_labels[i] == label]
            if local_train_sizes_temp:
                local_weights_temp = [model for i, model in enumerate(local_weights) if self.cluster_labels[i] == 0]
                ratio_temp = (np.array(local_train_sizes_temp) / np.sum(local_train_sizes_temp)).tolist()
                # self.set_weights(self.model, self.aggregate(local_weights_temp, ratio_temp))
                self.model_parameters[label + 1] = self.aggregate(local_weights_temp, ratio_temp)
        # local_train_sizes_1 = [local_train_sizes[i] for i in range(len(local_train_sizes)) if
        #                        self.cluster_labels[i] == 1]
        # if local_train_sizes_1:
        #     local_weights_1 = [model for i, model in enumerate(local_weights) if self.cluster_labels[i] == 1]
        #     ratio_1 = (np.array(local_train_sizes_1) / np.sum(local_train_sizes_1)).tolist()
        #     self.set_weights(self.model1, self.aggregate(local_weights_1, ratio_1))
        # self.logger.print(f'global model has been updated ({time.time() - st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self, ):
        model_dict = {f'model_{i}': model for i, model in enumerate(self.model_parameters)}
        model_dict['cluster_labels'] = self.cluster_labels
        return model_dict

        # def save_state(self, model):
    #     torch_save(self.args.checkpt_path, 'server_state.pt', {
    #         'model': get_state_dict(model),
    #     })
