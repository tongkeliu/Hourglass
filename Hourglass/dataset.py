from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        super(DatasetSplit,self).__init__()
        self.dataset = dataset
        self.idxs = idxs
    
    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image, label
    
    def __len__(self):
        return len(self.idxs)

def get_dataset(args):
    if args.dataset == 'cifar10':
        data_dir = '../data/cifar'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root = data_dir, train = True, transform = transform, download=True)
        test_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)

    elif args.dataset == 'cinic10':
        data_dir = '../data/cinic'
        
    return train_dataset, test_dataset

def dataset_iid(dataset, num_users):
    one_part = len(dataset) // num_users
    idxs = [i for i in range(len(dataset))]
    user_idxs = {}
    for i in range(num_users):
        user_idxs[i] = np.random.choice(idxs, one_part, replace=False)
        idxs = list(set(idxs)-set(user_idxs[i]))
    return user_idxs