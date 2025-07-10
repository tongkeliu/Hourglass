from torchvision.models import resnet50
import torch.nn as nn
from torch.nn import *
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from torch import Tensor
net = resnet50()

class ClientModelResNet50(nn.Module):
    def __init__(self):
        super(ClientModelResNet50, self).__init__()
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x

class ServerModelResNet50(nn.Module):
    def __init__(self, out_feature):
        super(ServerModelResNet50, self).__init__()
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool
        self.fc = nn.Linear(net.fc.in_features, out_feature, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ClientModelVgg16(nn.Module):
    def __init__(self) -> None:
        super(ClientModelVgg16, self).__init__()
        
        self.layer1 = nn.Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.layer2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ServerModelVgg16(nn.Module):
    def __init__(self, num_classes) -> None:
        super(ServerModelVgg16, self).__init__()
        self.layer3 = nn.Sequential(
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.layer4 = nn.Sequential(
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.layer5 = nn.Sequential(
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.avg_pool = AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            Linear(in_features=25088, out_features=4096, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            Linear(in_features=4096, out_features=4096, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            Linear(in_features=4096, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
    
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

class ClientModelViT(nn.Module):
    '''PatchEmbedding part'''
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
 
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class ServerModelViT(nn.Module):
    def __init__(self, emb_size = 768, depth = 12, n_classes = 1000, **kwargs):
        super().__init__()
        self.encoder = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

class ClientModelCharCNN(nn.Module):
    def __init__(self, num_features) -> None:
        super(ClientModelCharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )     
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ServerModelCharCNN(nn.Module):
    def __init__(self, num_classes):
        super(ServerModelCharCNN, self).__init__()          
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()    
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class ClientModelLSTM(nn.Module):
    def __init__(self, input_size) -> None:
        super(ClientModelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, num_layers=1, batch_first=True)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return x

class ServerModelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_classes=4, num_layers=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, num_classes)
 
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, (h, c) = self.lstm(x)
        x = self.forwardCalculation(h[-1])
        return x


def get_model(model, client_device, num_classes):
    if model == 'resnet50':
        client_side_model = ClientModelResNet50().to(client_device)
        server_side_model = ServerModelResNet50(num_classes)
    elif model == 'vgg16':
        client_side_model = ClientModelVgg16().to(client_device)
        server_side_model = ServerModelVgg16(num_classes)
    elif model == 'vit':
        client_side_model = ClientModelViT().to(client_device)
        server_side_model = ServerModelViT(n_classes=num_classes)
    elif model == 'charcnn':
        client_side_model = ClientModelCharCNN(70).to(client_device)
        server_side_model = ServerModelCharCNN(num_classes=num_classes)
    elif model == 'lstm':
        client_side_model = ClientModelLSTM(70).to(client_device)
        server_side_model = ServerModelLSTM(32, num_classes=num_classes)
    
    return client_side_model, server_side_model


if __name__ == "__main__":
    from torch.optim import SGD, Adam
    from torch.utils.data import DataLoader
    from dataset import get_dataset
    from tqdm import tqdm

    train_dataset, test_dataset = get_dataset('agnews')
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    device = 'cuda'
    model = ServerModelLSTM(70, num_classes=4).to(device)
    # optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.5)
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    train_acc, train_loss = [], []
    for i in range(5):
        for idx, (data, label) in enumerate(tqdm(train_dataloader)):
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = torch.sum(torch.argmax(pred, dim=1) == label).data / len(pred)
            train_acc.append(acc)
            train_loss.append(loss.item())
        
            if idx % 50 == 0 and idx != 0:
                print(f"acc:{sum(train_acc) / len(train_acc)} loss:{sum(train_loss) / len(train_loss)}")
                train_acc, train_loss = [], []