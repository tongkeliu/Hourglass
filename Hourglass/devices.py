import torch
from torch.optim import SGD
import torch.nn as nn
from dataset import DatasetSplit
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from copy import deepcopy
import numpy as np

class Client():
    def __init__(self, args, train_dataset, train_idxs, test_dataset, test_idxs, user_id):
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.user_id = user_id
        self.train_dataloader = DataLoader(DatasetSplit(train_dataset, train_idxs), batch_size=self.args.local_bs, shuffle=True)
        self.test_dataloader = DataLoader(DatasetSplit(test_dataset, test_idxs), batch_size=args.local_bs, shuffle=False)
    
    def forward(self, model):
        self.model = model
        self.optimizer = SGD(self.model.parameters(), self.args.local_lr, self.args.local_mtm)
        self.model = self.model.to(self.args.device)
        self.model.train()

        for i in range(self.args.local_ep):
            self.accs = []
            for data, label in self.train_dataloader:
                data, label = data.to(self.args.device), label.to(self.args.device)
                self.intermediate_feature = self.model(data)
                feature = self.intermediate_feature.clone().detach().cpu()
                # scheduler.add_feature(feature, label, self.user_id)
                yield feature, label, self.user_id

    def backprop(self, server_grad, pred, label):

        # no need to calculate loss
        self.optimizer.zero_grad()
        self.intermediate_feature.backward(server_grad)
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
        self.optimizer = SGD(model.parameters(), self.args.local_lr, self.args.local_mtm)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, feature, label):

        self.model.train()
        pred = self.model(feature)
        loss = self.criterion(pred, label)

        self.optimizer.zero_grad()
        loss.backward()
        feature_grad = feature.grad.clone().detach()
        self.optimizer.step()

        return feature_grad, pred

    def evaluate(self, feature, label):
        self.model.eval()
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

    def schedule(self, clients, server, num):
        cluster = self.clusterer(n_clusters=self.args.n_clusters, \
            max_iter=100, random_state=self.args.seed)
        features = [self.user_feature[user][0].detach().numpy() for user in self.user_feature]
        features = np.array(features).reshape(len(features),-1)
        cluster.fit(features)
        pred = cluster.predict(features)

        backward_order = self.order_features(pred)
        client_weight_list = []
        for user_id in backward_order:
            feature = self.user_feature[user_id][0].to(self.args.device).requires_grad_(True)
            server_grad, pred = server.train(feature, self.user_feature[user_id][1])
            weight = clients[user_id].backprop(server_grad, pred, self.user_feature[user_id][1])
            client_weight_list.append(weight)
        
        return client_weight_list

    def add_feature(self, feature, label, user_id):
        self.user_feature[user_id] = (feature, label)
    
    def order_features(self, pred):
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
                