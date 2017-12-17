import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
import torch

label_path = os.path.join(os.getcwd(), '../data/raw/t10k-labels-idx1-ubyte')
image_path = os.path.join(os.getcwd(), '../data/raw/t10k-images-idx3-ubyte')

lab_file = open(label_path, 'rb')
labels = np.frombuffer(lab_file.read(), dtype=np.uint8,
                       offset=8)


img_file = open(image_path, 'rb')
images = np.frombuffer(img_file.read(), dtype=np.uint8,
                       offset=16).reshape(10000, 28, 28)

dic = {0: 'T-shirt/top',
       1: 'Trouser',
       2: 'Pullover',
       3: 'Dress',
       4: 'Coat',
       5: 'Sandal',
       6: 'Shirt',
       7: 'Sneaker',
       8: 'Bag',
       9: 'Ankle boot'}

plt.title(dic[labels[1]])
plt.imshow(images[1], cmap='gray')
plt.savefig('pullover.png')


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
        return F.log_softmax(x)

model = Net()
model.load_state_dict(torch.load('best_model_mnist.pth'))

parameter = list(model.parameters())
print(parameter)