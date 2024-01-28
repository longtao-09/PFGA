from misc.utils import *


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.n_workers = 1
        self.client_id = None

        from torch_geometric.loader import DataLoader
        self.DataLoader = DataLoader
        self.client_pa, self.client_par = [], []
        self.all_data = None
        self.get_all_data()

    def switch(self, client_id):
        if not self.client_id == client_id:
            self.client_id = client_id
            # self.partition = get_data(self.args, client_id=client_id)
            # self.pa_loader = self.DataLoader(dataset=self.partition, batch_size=1,
            #                                  shuffle=False, num_workers=self.n_workers, pin_memory=True)
            self.partition = self.client_par[client_id]
            self.pa_loader = self.client_pa[client_id]
            self.train_size = torch.sum(self.partition[0].train_mask).item()

    def get_all_data(self):
        for client_id in range(self.args.n_clients):
            partition = get_data(self.args, client_id=client_id)
            pa_loader = self.DataLoader(dataset=partition, batch_size=1,
                                        shuffle=False, num_workers=self.n_workers, pin_memory=False)
            self.client_par.append(partition)
            self.client_pa.append(pa_loader)
        self.all_data = torch_load(
            self.args.data_path,
            f'{self.args.dataset}_all/{self.args.dataset}_all.pt'
        )['data']


def get_data(args, client_id):
    # print(args.data_path+'sdasdasd')
    return [
        torch_load(
            args.data_path,
            f'{args.dataset}_{args.mode}/{args.n_clients}/partition_{client_id}.pt'
        )['client_data']
    ]
