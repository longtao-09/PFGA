import os
import statistics

from parser import Parser
from datetime import datetime

from misc.utils import *
# from modules.multiprocs import ParentProcess
from modules.multiprocs_new import ParentProcess
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")


def main(args):
    args = set_config(args)
    if args.model == 'fedavg':
        from models.fedavg.server import Server
        from models.fedavg.client import Client
    elif args.model == 'fedpub':
        from models.fedpub.server import Server
        from models.fedpub.client import Client
    elif args.model == 'fednapl':
        from models.fednapl.server import Server
        from models.fednapl.client import Client
    elif args.model == 'fedacg':
        from models.fedacg.server import Server
        from models.fedacg.client import Client
        # args.alpha, args.beta, args.gamma = 0.7772, 1.8951, 2.058
    elif args.model == 'fedpfl':
        from models.fedpfl.server import Server
        from models.fedpfl.client import Client
    elif args.model == 'fedprox':
        from models.fedprox.server import Server
        from models.fedprox.client import Client
    elif args.model == 'fedpacg':
        from models.fedpacg.server import Server
        from models.fedpacg.client import Client
    elif args.model == 'fedcacg':
        from models.fedcacg.server import Server
        from models.fedcacg.client import Client
        args.alpha, args.beta, args.gamma = 0.7772, 1.8951, 2.058
    elif args.model == 'fedmacg':
        from models.fedmacg.server import Server
        from models.fedmacg.client import Client
        args.alpha, args.beta, args.gamma = 0.7772, 1.8951, 2.058
    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)

    print(args)
    pp = ParentProcess(args, Server, Client)
    pp.start()

    # num = 10
    # test_acc_list = []
    # for i in range(num):
    #     pp = ParentProcess(args, Server, Client)
    #     last_test_acc = pp.start()
    #     test_acc_list.append(last_test_acc)
    # avg_test_acc = statistics.mean(test_acc_list)
    # std_acc = np.std(np.array(test_acc_list))
    # print(f'train num:{num - 1},'
    #       f'avg test acc: {avg_test_acc * 100:.1f}%, std acc:{std_acc * 100:.2f},')


def set_config(args):
    args.base_lr = 1e-3
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    # args.weight_decay = 1e-6

    # args.lr = 1e-3
    # args.base_lr = 1e-3
    # args.min_lr = 1e-5
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
    main(Parser().parse())
