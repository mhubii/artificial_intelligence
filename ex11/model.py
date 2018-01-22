import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def weight_init(self, mean=0., var=0.075**2):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, var)
                m.bias.data.normal_(mean, var)


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

    def weight_init(self, mean=0., var=0.075**2):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, var)
                m.bias.data.normal_(mean, var)
