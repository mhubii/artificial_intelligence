import numpy as np
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from model import VAE
from code import apply_tsne_img

# Load MNIST handwritten digits data set.
data_set = datasets.MNIST('data',
                          train=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]),
                          download=True)

# Create model.
vae = VAE(28*28)
vae.load_state_dict(torch.load('vae_10_epochs.pth'))

# Encode.
means = np.empty([400, 64])
imgs = np.empty([400, 28, 28])

for i in range(400):
    img = data_set[i][0]
    imgs[i] = img
    img = Variable(img).view(1, -1)

    mean = vae.forward_mean(img).data.numpy()
    means[i] = mean

apply_tsne_img(means, imgs)

"""
# Cluster.
k_means = KMeans(n_clusters=10).fit(means)
centers = k_means.cluster_centers_

plt.imshow(centers)
plt.show()
"""