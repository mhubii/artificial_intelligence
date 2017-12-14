import torch
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
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# train the model
for epoch in range(100):
    for i, sample in enumerate(train_loader):
        # extract features and label from sample
        feature = sample['x']
        label = sample['y']

        # forward -> optimize -> backward
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(feature)
        loss = criterion(feature, sample)
        loss.backward()
        optimizer.step()

torch.save(net.state_dict(), 'net_one_trained.pk1')


