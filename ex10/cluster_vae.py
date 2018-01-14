import numpy as np
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from model import VAE
from code import apply_tsne_img


def cluster_center_img():
    # Cluster.
    k_means = KMeans(n_clusters=10, random_state=1).fit(means)
    centers = k_means.cluster_centers_

    fig = plt.figure()
    fig.suptitle('Cluster Center Images')

    for i in range(10):
        # Get center and pass through vae.
        center = torch.from_numpy(centers[i])
        center = Variable(center).float()
        out = vae.forward_decoder(center)

        plt.subplot('25{}'.format(i))
        plt.imshow(out.data.numpy().reshape([28, 28]))
        plt.axis('off')

    plt.savefig('img/cluster_center_images')


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
log_vars = np.empty([400, 64])
imgs = np.empty([400, 28, 28, 1])

for i in range(400):
    img = data_set[i][0]
    imgs[i] = np.transpose(img, (1, 2, 0))
    img = Variable(img).view(1, -1)

    mean = vae.forward_mean(img).data.numpy()
    means[i] = mean

if __name__ == '__main__':
    # apply_tsne_img(means, imgs)
    cluster_center_img()

