import torch.optim as opt
import torch.nn as nn
import utils
import net_one

# data loader
loc = 'data/tripple_junction_data_training.txt'
train_loader = utils.ToyDataset(loc)

# model
net = net_one.Net()

# loss and optimizer
criterion = nn.MSELoss()
optimizer = opt.SGD(net.parameters(), lr=0.01)

# train the model
for epoch in range(100):
    for i, (x, y) in enumerate(train_loader):
        # forward -> optimize -> backward
        pass

