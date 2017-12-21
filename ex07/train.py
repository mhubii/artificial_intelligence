import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils
import net_one

# data loader
loc = 'data/triple_junction_data_training.txt'
train_loader = utils.ToyDataset(loc)

# model
net = net_one.Net()

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# train the model
for epoch in range(1000):
    for i, sample in enumerate(train_loader):
        # extract features and label from sample. Convert to torch variables.
        features = Variable(torch.from_numpy(sample['features']).float())
        label = Variable(torch.FloatTensor([sample['labels']]))

        # forward -> optimize -> backward
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(features)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

torch.save(net.state_dict(), 'trained_net/net_one_trained.pk1')
