import argparse
import copy
import torch
import logging

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.6,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")
    parser.add_argument('--local_lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--local_mtm', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--model', type=str, default='resnet50', help='model to use')
    parser.add_argument('--device', type=str, default='cuda', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--strategy', type=str, default='DFF', help='feature scheduleing\
                        strategys', choices=['FCFS', 'SFF', 'DFF'])
    parser.add_argument('--n_clusters', type=int, default=4, help='clusters for kmeans')
    parser.add_argument('--split_method', type=str, default='LSH', help='method used\
                        to ensure data heterogeneity')

    args = parser.parse_args()
    return args

def get_logger():
    logger = logging.getLogger('multi_logger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('output.log')
    file_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
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