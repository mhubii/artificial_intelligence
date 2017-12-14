from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_dataset(txt_filepath):
    """Loads a toy dataset from a text file.
    
    Args:
        txt_filepath: file path to the text file containing the dataset
    Returns:
        data: NxD ndarray, where N is the number of samples in the dataset and D is the dimension of each sample 
        target: N-dimensional vector, where N is the number of samples in the dataset
    """
    coords_x,coords_y,labels = np.loadtxt(txt_filepath).T
    data = np.array([coords_x,coords_y]).T
    # labels should starts from 0
    target = labels - 1 
    return data, target


def visualize_toy_dataset(data, target):
    """Plots a 2D toy dataset with 3 classes.
    
    Args:
        data: NxD ndarray, where N is the number of samples in the dataset and D is the dimension of each sample 
        target: N-dimensional vector, where N is the number of samples in the dataset
    """
    n = data.shape[0]
    for i in range(0, n):
        if target[i] == 0:
            plt.plot(data[i, 0], data[i, 1], 'ro')
        elif target[i] == 1:
            plt.plot(data[i, 0], data[i, 1], 'go')
        else:
            plt.plot(data[i, 0], data[i, 1], 'bo')


def create_grid_coords(N=50):
    """ Creates a 2D grid centered on the origin with edge length 2.

    Note: Using the generated grid as input to your network, you can visualize the learned class boundaries.
    For plotting you can use imshow(...) from matplotlib.

    Returns:
        N^2x2 ndarray, where each row correspondes to a pair of coordinates in 2d space
    """
    xx=-(1-2*np.array(range(0,N))/float(N-1))
    grid_x, grid_y=np.meshgrid(xx,xx)
    grid=np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    return grid


class ToyDataset(Dataset):
    """ A toy dataset class which implements the abstract class torch.utils.data.Dataset .
    (for reference see http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset)
    """
    def __init__(self):
        pass

    def __getitem__(self, index):
        return NotImplemented

    def __len__(self):
        return NotImplemented
