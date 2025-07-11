from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import os
import csv
import json
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio

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

class AGNEWs(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0 = 1014):
        """Create AG's News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        self.l0 = l0
        # read alphabet
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)
                    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def loadAlphabet(self, alphabet_path = './data/agnews/alphabet.json'):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase = True):
        self.label = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                self.label.append(int(row[0]) - 1)
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()                
                self.data.append(txt)

        self.y = torch.LongTensor(self.label)

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char)!=-1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples/float(self.label.count(c)) for c in label_set]    
        return class_weight, num_class


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, root_dir = './data'):
        super().__init__(root_dir, download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]
        if subset == "valid":
            self._walker = load_list("validation_list.txt")
        elif subset == "test":
            self._walker = load_list("testing_list.txt")
        elif subset == "train":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

class SpeechCommands(Dataset):
    def __init__(self, subset, root_dir):
        super(SpeechCommands, self).__init__()
        self.dataset = SubsetSC(subset, root_dir)
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.dataset)))
        self.transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)

    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
        waveform = self.transform(waveform)
        label = torch.tensor(self.labels.index(label))
        return waveform, label
    
    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    tensors, targets = [], []
    for waveform, label in batch:
        tensors += [waveform.t()]
        targets += [label]
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.)
    tensors = tensors.permute(0,2,1)
    targets = torch.stack(targets)
    return tensors, targets

def get_collate_fn(args):
    if args.dataset == 'sc':
        return collate_fn

def get_dataset(dataset):
    if dataset == 'cifar10':
        data_dir = './data/cifar'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ])
        train_dataset = datasets.CIFAR10(root = data_dir, train = True, transform = transform, download=True)
        test_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    elif dataset == 'cinic10':
        data_dir = './data/cinic-10'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.478, 0.472, 0.430), (0.242, 0.238, 0.258))
        ])
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
    elif dataset == 'agnews':
        data_dir = './data/agnews'
        alphabet_path = './data/agnews/alphabet.json'
        train_dataset = AGNEWs(os.path.join(data_dir, 'train.csv'), alphabet_path)
        test_dataset = AGNEWs(os.path.join(data_dir, 'test.csv'), alphabet_path)
    elif dataset == 'sc':
        data_dir = './data'
        train_dataset = SpeechCommands('train', data_dir)
        test_dataset = SpeechCommands('test', data_dir)

    return train_dataset, test_dataset

def dataset_iid(dataset, num_users):
    one_part = len(dataset) // num_users
    idxs = [i for i in range(len(dataset))]
    user_idxs = {}
    for i in range(num_users):
        user_idxs[i] = np.random.choice(idxs, one_part, replace=False)
        idxs = list(set(idxs)-set(user_idxs[i]))
    return user_idxs