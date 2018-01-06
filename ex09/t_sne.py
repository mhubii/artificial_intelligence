from exercise_09_starter import Autoencoder
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load data set.
loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
    batch_size=500, shuffle=True)

# Load model.
model = Autoencoder()
model.load_state_dict(torch.load('best_model_autoencoder.pth'))

# Obtain encoded representation of samples.
data, label = next(iter(loader))
label = label.numpy()

data = Variable(data)
encoded_data = model.forward_encoder(data)
encoded_data = encoded_data.data.numpy()

# Use t-SNE to reduce dimensionality.
tsne = TSNE(n_components=2)
data_2d = tsne.fit_transform(encoded_data)

# Plot results.
color_map = cm.rainbow(np.linspace(0, 1, 10))

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

plt.figure()

for idx, lab in enumerate(np.unique(label)):
    plt.scatter(x=data_2d[label == lab, 0],
                y=data_2d[label == lab, 1],
                c=color_map[lab], label=dic[lab])

plt.title('t-SNE Visualization of Fashion MNIST')
plt.xlabel('x in t-SNE')
plt.ylabel('y in t-SNE')
plt.legend()
plt.savefig('img/t_sne_fashion_mnist.png')
