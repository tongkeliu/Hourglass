from torchvision.models import resnet50
import torch.nn as nn
import torch
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

if __name__ == "__main__":
    client_side_model = ClientModelResNet50()
    server_side_model = ServerModelResNet50(10)
    print(client_side_model)
    print(server_side_model)
    print(resnet50())