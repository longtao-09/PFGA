import time
import torch.nn.functional as F

from misc.utils import *
from data.loader import DataLoader
from modules.logger import Logger
from sklearn.metrics import roc_auc_score, average_precision_score


class ServerModule:
    def __init__(self, args, sd, gpu_server):
        self.args = args
        self._args = vars(self.args)
        self.gpu_id = gpu_server
        self.sd = sd
        self.logger = Logger(self.args, self.gpu_id, is_server=True)

    def get_active(self, mask):
        active = np.absolute(mask) >= self.args.l1
        return active.astype(float)

    def aggregate(self, local_weights, ratio=None):
        aggr_theta = OrderedDict([(k, None) for k in local_weights[0].keys()])
        if ratio is not None:
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name] * ratio[j] for j, theta in enumerate(local_weights)], 0)
        else:
            ratio = 1 / len(local_weights)
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
        return aggr_theta


class ClientModule:
    def __init__(self, args, w_id, g_id, sd):
        self.sd = sd
        self.gpu_id = g_id
        self.worker_id = w_id
        self.args = args
        self._args = vars(self.args)
        self.loader = DataLoader(self.args)
        self.logger = Logger(self.args, self.gpu_id)

    def switch_state(self, client_id):
        self.client_id = client_id
        self.loader.switch(client_id)
        self.logger.switch(client_id)
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def chance_state(self, client_state, client_id):
        self.client_id = client_id
        self.loader.switch(client_id)
        self.logger.switch(client_id)
        if client_state[client_id] != {}:
            self.load_state1(client_state[client_id])
        else:
            self.init_state()

    def is_initialized(self):
        return os.path.exists(os.path.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))

    @property
    def init_state(self):
        raise NotImplementedError()

    @property
    def save_state(self):
        raise NotImplementedError()

    @property
    def load_state(self):
        raise NotImplementedError()

    @property
    def load_state1(self, loaded):
        raise NotImplementedError()

    def loss_comp(self, x1, x2):
        cosine_similarity = F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1)
        similarity_diagonal = cosine_similarity.diagonal()
        similarity = cosine_similarity.fill_diagonal_(0).sum(dim=-1)
        loss = torch.log(similarity_diagonal / similarity).mean()
        if torch.isnan(loss):
            loss = 0
        return loss

    @torch.no_grad()
    def validate(self, mode='test', model=None):
        loader = self.loader.partition[0]
        # loader = self.loader.pa_loader
        with torch.no_grad():
            target, pred, loss = [], [], []
            # for _, batch in enumerate(loader):
            batch = loader
            batch = batch.cuda(self.gpu_id)
            mask = batch.test_mask if mode == 'test' else batch.val_mask
            y_hat, lss = self.validation_step(batch, mask, model)
            pred.append(y_hat[mask])
            target.append(batch.y[mask])
            loss.append(lss)
            acc = self.accuracy(torch.stack(pred).view(-1, self.args.n_clss), torch.stack(target).view(-1))
        return acc, np.mean(loss)

    @staticmethod
    def eval_edge_pred(adj_pred, val_edges, edge_labels):
        logits = adj_pred[val_edges]
        logits = np.nan_to_num(logits)
        roc_auc = roc_auc_score(edge_labels, logits)
        ap_score = average_precision_score(edge_labels, logits)
        return roc_auc, ap_score

    @torch.no_grad()
    def test_ep(self, model, data, train_edge):
        model.eval()
        adj = train_edge.train_pos_edge_index
        adj_logit = model(data, adj=adj)

        val_edges = torch.cat((train_edge.val_pos_edge_index, train_edge.val_neg_edge_index), axis=1).cpu().numpy()
        val_edge_labels = np.concatenate(
            [np.ones(train_edge.val_pos_edge_index.size(1)), np.zeros(train_edge.val_neg_edge_index.size(1))])

        test_edges = torch.cat((train_edge.test_pos_edge_index, train_edge.test_neg_edge_index), axis=1).cpu().numpy()
        test_edge_labels = np.concatenate(
            [np.ones(train_edge.test_pos_edge_index.size(1)), np.zeros(train_edge.test_neg_edge_index.size(1))])

        adj_pred = adj_logit.cpu()
        ep_auc, ep_ap = self.eval_edge_pred(adj_pred, val_edges, val_edge_labels)
        # print(f'EPNet train: auc {ep_auc:.4f}, ap {ep_ap:.4f}')

        ep_auc_test, ep_ap_test = self.eval_edge_pred(adj_pred, test_edges, test_edge_labels)
        # print(f'EPNet train,Final: auc {ep_auc_test:.4f}, ap {ep_ap:.4f}')

        return ep_auc, ep_ap, ep_auc_test, ep_ap_test

    @torch.no_grad()
    def validation_step(self, batch, mask=None, model=None):
        self.model.eval()
        if model == None:
            y_hat = self.model(batch)  # modif
        else:
            model.eval()
            y_hat = model(batch)  # modif
        if torch.sum(mask).item() == 0: return y_hat, 0.0
        lss = F.cross_entropy(y_hat[mask], batch.y[mask])
        return y_hat, lss.item()

    @torch.no_grad()
    def accuracy(self, preds, targets):
        if targets.size(0) == 0: return 1.0
        with torch.no_grad():
            preds = preds.max(1)[1]
            acc = preds.eq(targets).sum().item() / targets.size(0)
        return acc

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'args': self._args,
            'log': self.log
        })

    def get_optimizer_state(self, optimizer):
        state = {}
        for param_key, param_values in optimizer.state_dict()['state'].items():
            state[param_key] = {}
            for name, value in param_values.items():
                if torch.is_tensor(value) == False: continue
                state[param_key][name] = value.clone().detach().cpu().numpy()
        return state
