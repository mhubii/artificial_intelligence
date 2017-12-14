import torch
import utils
import matplotlib.pyplot as plt

# load trained net
net = torch.load('net_one_trained.pk1')

# create grid
grid = utils.create_grid_coords(N=50)

# predict on grid
prediction = net(grid)
plt.imshow(prediction)
