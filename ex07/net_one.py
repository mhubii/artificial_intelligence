import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU()
        )
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out
