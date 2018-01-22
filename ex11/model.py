import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __int__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def weights_init(self):
        self.fc1.weight.data.normal_(0., 0.075**2)
        self.fc1.bias.data.normal_(0., 0.075**2)
        self.fc2.weight.data.normal_(0., 0.075**2)
        self.fc2.bias.data.normal_(0., 0.075 ** 2)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def weights_init(self):
        self.fc1.weight.data.normal_(0, 0.075**2)
        self.fc1.bias.data.normal_(0., 0.075 ** 2)
        self.fc2.weight.data.normal_(0, 0.075**2)
        self.fc2.bias.data.normal_(0., 0.075 ** 2)


