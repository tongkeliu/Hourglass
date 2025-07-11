from tqdm import tqdm
import torch
import random
import numpy as np

from utils import *
from model import *
from devices import Client, Server, Scheduler
from dataset import get_dataset, dataset_iid, get_collate_fn
from copy import deepcopy
from time import time


def same_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def main(args):
    train_dataset, test_dataset = get_dataset(args.dataset)
    client_side_model, server_side_model = get_model(args.model, args.device, args.num_classes)
    user_idxs = dataset_iid(train_dataset, args.num_users)
    user_idxs_test = dataset_iid(test_dataset, args.num_users)

    logger = get_logger(args)
    logger.info(args)
    writer = Writer(args)
    collate_fn = get_collate_fn(args)
    logger.info(f"we use {min(args.M_GPU, torch.cuda.device_count())} GPUs")
    servers = [Server(deepcopy(server_side_model).to(f'cuda:{i}'), args) for i in range(args.M_GPU)]
    # server = Server(server_side_model, args)
    scheduler = Scheduler(args, args.strategy)
    current_time = 0

    for epoch in range(args.epochs):
        num = max(int(args.frac * args.num_users), 1)
        users = np.random.choice(args.num_users, num, replace=False)
        clients, client_feature_generator = [], []

        for user_id, user in enumerate(users):
            client = Client(args, train_dataset, user_idxs[user], test_dataset,\
                            user_idxs_test[user], user_id, collate_fn)
            clients.append(client)
            client_feature_generator.append(client.forward(model=deepcopy(client_side_model)))

        try:
            while True:
                for generator in client_feature_generator:
                    scheduler.add_feature(*next(generator))
                client_weight_list = scheduler.schedule(clients, servers, num) #
        except StopIteration:
            logger.debug("a global epoch finished")

        avg_weight = FedAvg(client_weight_list)
        client_side_model.load_state_dict(avg_weight)
        server_weights = [server.model.to('cpu').state_dict() for server in servers]
        avg_weight = FedAvg(server_weights)
        server_side_model.load_state_dict(avg_weight)
        for i in range(len(servers)):
            servers[i] = Server(deepcopy(server_side_model).to(f'cuda:{i}'), args)
        
        # evaluation part
        server = available_server(servers)
        client = Client(args, train_dataset, user_idxs[users[0]], test_dataset, \
                        user_idxs_test[users[0]], 0, collate_fn)
        acc, loss = client.evaluate(model=deepcopy(client_side_model), server=server) #

        # record the evaluation result
        logger.info("epoch:{} accs:{} loss:{}".format(epoch, acc, loss))
        writer.add_scalars("epoch", {"acc":acc, "loss":loss}, epoch)

        current_time = current_time + scheduler.duration
        scheduler.duration = 0
        writer.add_scalars("time", {"acc":acc}, float(current_time))
        logger.debug("time:{} accs:{} loss:{}".format(current_time, acc, loss))


if __name__ == "__main__":
    args = args_parser()
    same_seed(args.seed)

    main(args)