from lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# initialize and load the trained network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


classifier = Net()
classifier.load_state_dict(torch.load('best_model_mnist.pth'))

# select the target class
target_class = 1

# fix the network weights
for param in classifier.parameters():
    param.requires_grad = False

# means and std used for the image normalization
mean = 0.1307
std = 0.3081

# allocate memory for the image variable
# NOTE: we need the gradients with respect to the input images
# This requires certain changes in the initialization of the Pytorch Variable
imagevar = Variable(torch.zeros((1, 1, 28, 28)).type(torch.FloatTensor), requires_grad=True)

# calculate the gradients of the objective function
# notice that we will use gradient descend algorithm below, so we need to change the
# sign of the objective function
grad = Variable(torch.zeros(1, 10).type(torch.FloatTensor))
output = classifier(imagevar)[0, target_class]
output.retain_grad()
output.backward()

grad[0, target_class] = -output.grad

# set learning parameters of the gradient descend
LR = 0.5  # worked well for me
NUM_ITER = 10000
# start with a black image
# initialized

# you may use SGD optimizer from pytorch or implement the update step by yourself
optimizer = torch.optim.SGD([imagevar], lr=LR, momentum=0.9)
# scheduler track the changes of the objective function and reduces LR when needed
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, min_lr=0.5e-6)

for i in range(NUM_ITER):
    print('[{0:05d}/{1:d}]'.format(i+1, NUM_ITER), end='\r')

    # set gradients of the optimizer to zero
    optimizer.zero_grad()

    # obtain the softmax avtivations of the last layer
    act_value = classifier(imagevar)
    # backpropagate to image domain
    act_value.backward(grad)

    # transmit the current value of the objective function to the scheduler
    scheduler.step(act_value.data[0, target_class-1])
    # make step toward the negative gradients:
    optimizer.step()

    # we clip the values of the updated image to the feasible region
    imagevar.data = torch.clamp(imagevar.data, -mean/std, (1-mean)/std)

# show image
max_activation = imagevar.data.numpy().reshape([28, 28])
plt.imshow(max_activation)
plt.show()
