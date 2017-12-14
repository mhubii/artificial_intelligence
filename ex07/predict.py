import torch
import utils
import numpy as np
import matplotlib.pyplot as plt
import net_one
import utils

# build net
net = net_one.Net()

# load trained net
net.load_state_dict(torch.load('trained_net/net_one_trained.pk1'))

# create grid
grid = utils.create_grid_coords(N=50)

# predict on grid
prediction = np.empty(50*50)
i = 0

for element in grid:
    element = torch.autograd.Variable(torch.from_numpy(element).float())
    prediction[i] = net(element)
    i += 1

# visualize toy data set
data, target = utils.load_dataset('data/triple_junction_data_training.txt')
data += 1
data *= 25
utils.visualize_toy_dataset(data, target)

# visualize prediction
prediction = prediction.reshape([50, 50])
plt.imshow(prediction)
plt.show()
