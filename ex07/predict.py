import torch
import utils

net = torch.load('net_one_trained.pk1')

grid = utils.create_grid_coords(N=50)

net.()