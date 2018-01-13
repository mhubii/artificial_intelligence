import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from argparse import ArgumentParser
from tqdm import tqdm

import model
import code


def train():
    for img, _ in tqdm(data_loader):
        # Flatten
        img = img.view(args.batch_size, -1)
        img = Variable(img)

        adam.zero_grad()

        out = vae.forward(img)
        mean = vae.forward_mean(img)
        log_var = vae.forward_log_var(img)

        loss = code.loss_function(out, img, mean, log_var, args.batch_size, 28, 1)
        loss.backward()
        adam.step()


if __name__ == '__main__':
    # Some initial settings.
    parser = ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    # Load MNIST handwritten digits data set.
    data_set = datasets.MNIST('data',
                              train=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                                  ]),
                              download=True)

    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True)

    # Variational autoencoder.
    vae = model.VAE(28*28)

    # Optimizer.
    adam = Adam(vae.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train()

    # Save results.
    torch.save(vae.state_dict(), 'vae_{}_epochs.pth'.format(args.epochs))
