import torch
from torch.optim import SGD, Adam
import torch.nn as nn
from dataset import DatasetSplit
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from copy import deepcopy
import numpy as np
from collections import defaultdict
from utils import available_server
from time import time

class Client():
    def __init__(self, args, train_dataset, train_idxs, test_dataset, test_idxs, user_id, collate_fn = None):
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.user_id = user_id
        self.train_dataloader = DataLoader(DatasetSplit(train_dataset, train_idxs),\
            batch_size=self.args.local_bs, shuffle=True, collate_fn=collate_fn)
        self.test_dataloader = DataLoader(DatasetSplit(test_dataset, test_idxs),\
            batch_size=args.local_bs, shuffle=False, collate_fn=collate_fn)
    
    def forward(self, model):
        self.model = model
        if self.args.optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), self.args.local_lr, self.args.local_mtm)
        elif self.args.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), self.args.local_lr)
        self.model = self.model.to(self.args.device)
        self.model.train()

        for i in range(self.args.local_ep):
            self.accs = []
            for data, label in self.train_dataloader:
                start_time = time()
                data, label = data.to(self.args.device), label
                self.intermediate_feature = self.model(data)
                feature = self.intermediate_feature.clone().detach().cpu()
                # scheduler.add_feature(feature, label, self.user_id)
                yield feature, label, self.user_id, time() - start_time

    def backprop(self, server_grad, pred, label):

        self.optimizer.zero_grad()
        self.intermediate_feature.backward(server_grad.to(self.args.device))
        self.optimizer.step()

        self.accs.append(torch.sum(torch.argmax(pred, dim=1)==label).data / len(pred))
        # if self.args.verbose:
        #     print("user_id:{} local training accuracy:{}".\
        #                 format(self.user_id, sum(self.accs) / len(self.accs)))
        
        return self.model.state_dict()

    def evaluate(self, model, server):
        model = model.to(self.args.device)
        model.eval()

        acc_list, loss_list = [], []

        for data, label in self.test_dataloader:
            data, label = data.to(self.args.device), label.to(self.args.device)
            feature = model(data).clone().detach()
            acc, loss = server.evaluate(feature, label)

            acc_list.append(acc)
            loss_list.append(loss)
        
        return sum(acc_list)/len(acc_list), sum(loss_list)/len(loss_list)


class Server():
    def __init__(self, model, args):
        self.model = model
        self.args = args
        if self.args.optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), self.args.local_lr, self.args.local_mtm)
        elif self.args.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), self.args.local_lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, feature, label):
        server_device = next(self.model.parameters()).device
        self.model.train()

        feature, label = feature.to(server_device).requires_grad_(True), label.to(server_device)
        pred = self.model(feature)
        loss = self.criterion(pred, label)

        self.optimizer.zero_grad()
        loss.backward()
        feature_grad = feature.grad.clone().detach()
        self.optimizer.step()

        return feature_grad, pred.detach().cpu()

    def evaluate(self, feature, label):
        self.model.eval()
        server_device = next(self.model.parameters()).device
        feature, label = feature.to(server_device), label.to(server_device)
        with torch.no_grad():
            pred = self.model(feature)
            acc = torch.sum(torch.argmax(pred, dim=1)==label) / len(pred)
            loss = self.criterion(pred, label)
        return acc, loss


class Scheduler():
    def __init__(self, args, strategy):
        self.user_feature = dict()
        self.clusterer = KMeans
        self.args = args
        self.strategy = strategy
        self.user_time = {"client_forward":[], "server_time":[], "client_backward":[]}
        self.duration = 0

    def schedule(self, clients, servers, num):
        if self.args.split_method == 'kmeans':
            backward_order = self.order_features_cluster()
        else:
            backward_order = self.order_feature_LSH()
            
        client_weight_list = []
        for user_id in backward_order:
            feature = self.user_feature[user_id][0]
            server_start_time = time()
            server_grad, pred = available_server(servers).train(feature, self.user_feature[user_id][1])
            server_finish_time = time()
            weight = clients[user_id].backprop(server_grad, pred, self.user_feature[user_id][1])
            client_finish_time = time()
            
            client_weight_list.append(weight)
            self.user_time["server_time"].append(server_finish_time - server_start_time)
            self.user_time["client_backward"].append(client_finish_time - server_finish_time)

        self.duration = self.duration + min(self.user_time['client_forward']) + \
            min(self.user_time['server_time']) + max(self.user_time['client_backward'])
        self.user_time = {"client_forward":[], "server_time":[], "client_backward":[]}

        return client_weight_list

    def add_feature(self, feature, label, user_id, duration):
        self.user_feature[user_id] = (feature, label)
        self.user_time["client_forward"].append(duration)
    
    def order_features_cluster(self):
        cluster = self.clusterer(n_clusters=self.args.n_clusters, \
            max_iter=100, random_state=self.args.seed)
        features = [self.user_feature[user][0].detach().numpy() for user in self.user_feature]
        features = np.array(features).reshape(len(features),-1)
        cluster.fit(features)
        pred = cluster.predict(features)

        backward_order = []
        bucket = dict()
        for pseudo_label in range(self.args.n_clusters):
            bucket[pseudo_label] = [idx for idx in range(len(pred)) if pred[idx] == pseudo_label]
            backward_order.extend(bucket[pseudo_label])

        if self.strategy == 'FCFS':
            backward_order = [i for i in range(len(pred))]
        elif self.strategy == 'SFF':
            pass
        else:
            backward_order = []
            while True:
                for pseudo_label in range(self.args.n_clusters):
                    if bucket[pseudo_label]:
                        backward_order.append(bucket[pseudo_label].pop(0))
                if len(backward_order) == len(pred):
                    break

        return backward_order
    
    def order_feature_LSH(self):
        n_bits = 2 # four hyperplane
        dim = len(self.user_feature[0][0].contiguous().view(-1))

        buckets = defaultdict(list)
        plane_norms = np.random.rand(n_bits, dim) - 0.5
        for user_id, (feature, label) in self.user_feature.items():
            dot = np.dot(feature.contiguous().view(-1), plane_norms.T)
            dot = dot > 0
            hash_str = ''.join(dot.astype(int).astype(str))
            buckets[hash_str].append(user_id)
        
        backward_order = []
        if self.strategy == 'FCFS':
            backward_order = [i for i in range(len(self.user_feature))]
        elif self.strategy == 'SFF':
            for hash_str, user_ids in buckets.items():
                backward_order.extend(user_ids)
        elif self.strategy == 'DFF':
            while True:
                for hash_str ,user_ids in buckets.items():
                    if user_ids:
                        backward_order.append(buckets[hash_str].pop(0))
                if len(self.user_feature) == len(backward_order):
                    break
        
        return backward_order
