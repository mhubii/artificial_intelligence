import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from model import Generator


def find_nearest_neighbours(samples, img):
    # Minimal loss and nearest neighbours.
    min_loss = np.full(4, float('inf'))
    nearest_neighbours = np.empty([4, 28, 28])

    # Search nearest neighbours of img.
    for s in samples:
        loss = np.linalg.norm(s-img)
        max_loss_idx = np.argmax(min_loss)

        if loss < min_loss[max_loss_idx]:
            min_loss[max_loss_idx] = loss
            nearest_neighbours[max_loss_idx] = s

    return nearest_neighbours


if __name__ == '__main__':
    # Load model.
    gen = Generator()
    gen.load_state_dict(torch.load('gen_state_dict.pth'))

    # Generate fake images.
    z = torch.randn(2000, 100)
    z = Variable(z)

    fake = gen(z)
    fake = fake.data.numpy().reshape(2000, 28, 28)

    # Real images.
    data_set = datasets.MNIST('data',
                              train=False,
                              download=True)

    idx = 0
    real = np.array(data_set[idx][0])
    print(data_set[idx][1])

    nearest_neighbours = find_nearest_neighbours(fake, real)

    # Save results.
    plt.figure()

    plt.subplot(151)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(real)

    for i in range(4):
        plt.subplot('15{}'.format(i + 2))
        plt.title('Neighbour')
        plt.axis('off')
        plt.imshow(nearest_neighbours[i])

    plt.savefig('img/nearest_neighbours.png')