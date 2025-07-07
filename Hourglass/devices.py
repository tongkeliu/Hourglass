import torch
from torch.optim import SGD
import torch.nn as nn
from dataset import DatasetSplit
from torch.utils.data import DataLoader

class Client():
    def __init__(self, args, train_dataset, train_idxs, test_dataset, test_idxs):
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.train_dataloader = DataLoader(DatasetSplit(train_dataset, train_idxs), batch_size=self.args.local_bs, shuffle=True)
        self.test_dataloader = DataLoader(DatasetSplit(test_dataset, test_idxs), batch_size=args.local_bs, shuffle=False)
    
    def train(self, model, server):
        optimizer = SGD(model.parameters(), self.args.local_lr, self.args.local_mtm)
        model = model.to(self.args.device)
        model.train()

        for i in range(self.args.local_ep):
            accs = []
            for data, label in self.train_dataloader:
                data, label = data.to(self.args.device), label.to(self.args.device)
                intermediate_feature = model(data)
                feature = intermediate_feature.clone().detach().requires_grad_(True)
                server_grad, pred = server.train(feature, label)
                # no need to calculate loss
                optimizer.zero_grad()
                intermediate_feature.backward(server_grad)
                optimizer.step()

                accs.append(torch.sum(torch.argmax(pred, dim=1)==label).data / len(pred))
            if self.args.verbose:
                print("local training accuracy:{}".format(sum(accs) / len(accs)))
        
        return model.state_dict()

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
            torch.cuda.empty_cache()
        
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