import os

import optuna

from parser import Parser
from datetime import datetime

from misc.utils import *
# from modules.multiprocs import ParentProcess
from modules.multiprocs_new import ParentProcess


def main(trial=None):
    args = Parser().parse()
    args = set_config(args)

    if args.model == 'fedavg':
        from models.fedavg.server import Server
        from models.fedavg.client import Client
        # elif args.model == 'fedpub':
        #     from models.fedpub.server import Server
        #     from models.fedpub.client import Client
        # elif args.model == 'fednapl':
        #     from models.fednapl.server import Server
        #     from models.fednapl.client import Client

        # elif args.model == 'fedpfl':
        #     from models.fedpfl.server import Server
        #     from models.fedpfl.client import Client
        # elif args.model == 'fedprox':
        #     from models.fedprox.server import Server
        #     from models.fedprox.client import Client
        # elif args.model == 'fedpacg':
        #     from models.fedpacg.server import Server
        #     from models.fedpacg.client import Client
    elif args.model == 'fedpfgaa':
        from models.fedpfgaa.server import Server
        from models.fedpfgaa.client import Client
    elif args.model == 'fedpfgac':
        from models.fedpfgac.server import Server
        from models.fedpfgac.client import Client
    elif args.model == 'fedpfgam':
        from models.fedpfgam.server import Server
        from models.fedpfgam.client import Client
    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)
    if trial is not None:
        args.alpha = trial.suggest_float('alpha', 0, 1)
        args.beta = trial.suggest_float('beta', 0, 10)
        args.gamma = trial.suggest_float('gama', 0, 5)
        # gamma = 0
    else:
        args.alpha, args.beta, args.gamma = 0.6900000000000001, 8.85, 3.6,  # 'acc': 0.8301999999999999
    print(args)
    pp = ParentProcess(args, Server, Client)
    # acc = pp.start()
    # if acc >= 0.826:
    #     print('!!!!!!!!!!!!!!!!!!!success!!!!!!!!!!!!!!!!!!!')
    #     exit()
    return pp.start()


def set_config(args):
    args.base_lr = 1e-3
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    # args.weight_decay = 1e-6

    # args.lr = 1e-1
    # args.base_lr = 1e-1
    # args.min_lr = 1e-1
    # args.momentum_opt = 0.9
    args.weight_decay = 5e-4

    args.warmup_epochs = 10
    args.base_momentum = 0.99
    args.final_momentum = 1.0

    args.n_dims = 64

    if args.dataset == 'Cora':
        args.n_feat = 1433
        args.n_clss = 7
        args.n_clients = 10 if args.n_clients is None else args.n_clients
        args.base_lr = 0.01 if args.lr is None else args.lr
    elif args.dataset == 'Citeseer':
        args.n_feat = 3703
        args.n_clss = 6
        args.n_clients = 10 if args.n_clients is None else args.n_clients
        args.base_lr = 0.01 if args.lr is None else args.lr
    elif args.dataset == 'Pubmed':
        args.n_feat = 500
        args.n_clss = 3
        args.n_clients = 10 if args.n_clients is None else args.n_clients
        args.base_lr = 0.01 if args.lr is None else args.lr
    elif args.dataset == 'coauthor-cs':
        args.n_feat = 6805
        args.n_clss = 15
        args.n_clients = 10 if args.n_clients is None else args.n_clients
        args.base_lr = 0.01 if args.lr is None else args.lr
    elif args.dataset == 'amazon-computers':
        args.n_feat = 767
        args.n_clss = 10
        args.n_clients = 10 if args.n_clients is None else args.n_clients
        args.base_lr = 0.01 if args.lr is None else args.lr
    elif args.dataset == 'amazon-photo':
        args.n_feat = 745
        args.n_clss = 8
        args.n_clients = 10 if args.n_clients is None else args.n_clients
        args.base_lr = 0.01 if args.lr is None else args.lr
    elif args.dataset == 'ogbn-arxiv':
        args.n_feat = 128
        args.n_clss = 40
        args.n_clients = 10 if args.n_clients is None else args.n_clients
        args.base_lr = 0.01 if args.lr is None else args.lr
    elif args.dataset == 'reddit':
        args.n_feat = 602
        args.n_clss = 41
        args.n_clients = 10 if args.n_clients is None else args.n_clients
        args.base_lr = 0.01 if args.lr is None else args.lr

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}'
    args.base_path = '.'
    args.data_path = f'{args.base_path}/datasets'
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    args.log_path = f'{args.base_path}/logs/{trial}'

    if args.debug is True:
        args.checkpt_path = f'{args.base_path}/debug/checkpoints/{trial}'
        args.log_path = f'{args.base_path}/debug/logs/{trial}'

    return args


if __name__ == '__main__':
    # main(Parser().parse())
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=500)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
