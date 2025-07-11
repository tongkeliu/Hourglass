import argparse
import copy
import torch
import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=80,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.6,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100,
                        help="local batch size: B")
    parser.add_argument('--local_lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--local_mtm', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        help='the optimizer for model', choices=['sgd', 'adam'])

    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset", choices=['cifar10','cinic10','agnews','sc'])
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes", choices=[10, 4, 35])
    parser.add_argument('--model', type=str, default='resnet50', help='model to use',\
                        choices=["resnet50", "vgg16", "vit", "charcnn", "lstm", "vgg16_1d"])

    parser.add_argument('--device', type=str, default='cuda:3', help="client side device")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    
    parser.add_argument('--strategy', type=str, default='SFF', help='feature scheduleing\
                        strategys', choices=['FCFS', 'SFF', 'DFF'])
    parser.add_argument('--n_clusters', type=int, default=4, help='clusters for kmeans')
    parser.add_argument('--split_method', type=str, default='LSH', help='method used\
                        to ensure data heterogeneity', choices=['kmeans', 'LSH'])
    parser.add_argument('--M_GPU', type=int, default=3, help='the number of gpu\
                        for server to train model')
    args = parser.parse_args()
    return args

def get_logger(args):
    base_dir = 'save/logs'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    details = f"{args.model}-{args.dataset}-{current_time}.log"
    log_dir = os.path.join(base_dir, details)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def get_GPU_info():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print("  Name:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print("  Properties:")
        print("    Compute capability:", props.major, props.minor)
        print("    Total memory:", props.total_memory / (1024 ** 2), "MB")
        print("    MultiProcessorCount:", props.multi_processor_count)

def available_server(servers):
    server = servers.pop(0)
    servers.append(server)

    return server

class Writer():
    def __init__(self, args, base_dir = './save/hourglass') -> None:
        self.args = args
        self.base_dir = base_dir
        self.prefix = f"{args.model}-{args.dataset}"
        self.writer = SummaryWriter(self.base_dir)
    
    def add_scalars(self, type, scalars: dict, step):
        self.writer.add_scalars(self.prefix+'/'+type+'_', scalars, step)
    
    def close(self):
        self.writer.close()