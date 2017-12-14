import utils
import torch.nn as nn

# define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv3 = nn.Conv2d(1, 3, kernel_size=5)
        self.fc1
        self.fc2

    def forward(self):
        pass
