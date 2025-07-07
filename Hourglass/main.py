from tqdm import tqdm
import torch
import random
import numpy as np

from utils import args_parser, FedAvg, get_logger
from model import *
from devices import Client, Server, Scheduler
from dataset import get_dataset, dataset_iid
from copy import deepcopy

def same_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def main(args, client_side_model, server_side_model):
    train_dataset, test_dataset = get_dataset(args)
    user_idxs = dataset_iid(train_dataset, args.num_users)
    user_idxs_test = dataset_iid(test_dataset, args.num_users)

    logger = get_logger()
    server = Server(server_side_model, args)
    scheduler = Scheduler(args, args.strategy)

    for epoch in range(args.epochs):
        num = max(int(args.frac * args.num_users), 1)
        users = np.random.choice(args.num_users, num, replace=False)
        clients, client_feature_generator = [], []

        for user_id, user in enumerate(users):
            client = Client(args, train_dataset, user_idxs[user], test_dataset, user_idxs_test[user], user_id)
            clients.append(client)
            client_feature_generator.append(client.forward(model=deepcopy(client_side_model)))

        try:
            while True:
                for generator in client_feature_generator:
                    scheduler.add_feature(*next(generator))
                client_weight_list = scheduler.schedule(clients, server, num)
        except StopIteration:
            avg_weight = FedAvg(client_weight_list)
            client_side_model.load_state_dict(avg_weight)

        client = Client(args, train_dataset, user_idxs[users[0]], test_dataset, user_idxs_test[users[0]], 0)
        acc, loss = client.evaluate(model=deepcopy(client_side_model), server=server)
        logger.info("epoch:{} accs:{} loss:{}".format(epoch, acc, loss))


if __name__ == "__main__":
    args = args_parser()
    same_seed(args.seed)

    if args.model == 'resnet50':
        client_side_model = ClientModelResNet50().to(args.device)
        server_side_model = ServerModelResNet50(args.num_classes).to(args.device)
    
    main(args, client_side_model, server_side_model)