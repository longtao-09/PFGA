import os
import sys
import time
import atexit
import numpy as np
from misc.utils import *
from models.nets import *
# from models.napl import *


class ParentProcess:
    def __init__(self, args, Server, Client):
        self.args = args
        self.gpus = [int(g) for g in args.gpu.split(',')]
        self.gpu_server = self.gpus[0]
        self.proc_id = os.getppid()
        print(f'main process id: {self.proc_id}')

        self.sd = {'is_done': False}
        self.create_workers(Client)
        self.server = Server(args, self.sd, self.gpu_server)

    def create_workers(self, Client):
        self.processes = []
        # gpu_id = self.gpus[worker_id] if worker_id <= len(self.gpus)-1 else self.gpus[worker_id%len(self.gpus)]
        gpu_id = self.gpus[0]
        self.client = WorkerProcess(self.args, gpu_id, self.sd, Client)

    def start(self):
        self.sd['is_done'] = False
        if os.path.isdir(self.args.checkpt_path) is False:
            os.makedirs(self.args.checkpt_path)
        if os.path.isdir(self.args.log_path) is False:
            os.makedirs(self.args.log_path)
        self.n_connected = round(self.args.n_clients * self.args.frac)
        for curr_rnd in range(self.args.n_rnds):
            self.curr_rnd = curr_rnd
            np.random.seed(self.args.seed + curr_rnd)
            self.selected = np.random.choice(self.args.n_clients, self.n_connected, replace=False).tolist()
            self.updated = set(self.selected)
            st = time.time()
            ##################################################
            self.server.on_round_begin(curr_rnd)
            ##################################################
            self.client.client_round(self.selected, curr_rnd)
            ###########################################
            self.server.on_round_complete(self.updated)
            ###########################################
            # print(f'[main] round {curr_rnd} done ({time.time() - st:.2f} s)')

        self.sd['is_done'] = True
        return self.server.last_test_acc


class WorkerProcess:
    def __init__(self, args, gpu_id, sd, Client, worker_id=0, q=None):
        self.q = q
        self.sd = sd
        self.args = args
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.is_done = False
        self.client = Client(self.args, self.worker_id, self.gpu_id, self.sd)
        self.client_state = defaultdict(dict)

    def client_round(self, id_list, curr_rnd):
        for client_id in id_list:
            ##################################
            # self.client.switch_state(client_id)
            self.client.chance_state(self.client_state, client_id)
            self.client.on_receive_message(curr_rnd)
            self.client.on_round_begin(client_id)
            self.client.update_state(self.client_state, client_id)

