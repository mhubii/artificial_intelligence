import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator, Discriminator


def train():
    for img, _ in tqdm(data_loader):
        # Resize.
        img = img.view(batch_size, -1)
        img = Variable(img).cuda()

        # Train discriminator.
        dis_opt.zero_grad()

        # Initialize labels.
        real_label = torch.ones(batch_size)
        real_label = Variable(real_label).cuda()

        fake_label = torch.zeros(batch_size)
        fake_label = Variable(fake_label).cuda()

        # Initialize noise.
        z = torch.randn(batch_size, 100)
        z = Variable(z).cuda()

        # Forward.
        dis_out = dis(img).squeeze()
        real_loss = mse_loss(dis_out, real_label)

        dis_out = dis(gen(z)).squeeze()
        fake_loss = mse_loss(dis_out, fake_label)

        # Backward.
        dis_loss = real_loss + fake_loss
        dis_loss.backward()
        dis_opt.step()

        # Train generator.
        gen_opt.zero_grad()

        # Initialize noise.
        z = torch.randn(batch_size, 100)
        z = Variable(z).cuda()

        # Forward.
        dis_out = dis(gen(z)).squeeze()
        gen_loss = mse_loss(dis_out, real_label)

        # Backward.
        gen_loss.backward()
        gen_opt.step()


if __name__ == '__main__':
    # Initial parameters.
    batch_size = 64

    # Load MNIST handwritten digits data set.
    data_set = datasets.MNIST('data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)

    # Initialize models and weights.
    gen = Generator().cuda()
    gen.weight_init()

    dis = Discriminator().cuda()
    dis.weight_init()

    # Optimizer and loss.
    gen_opt = Adam(gen.parameters(), lr=0.001)
    dis_opt = Adam(dis.parameters(), lr=0.001)

    mse_loss = nn.MSELoss().cuda()

    # Train.
    const_z = torch.randn(64, 100)
    const_z = Variable(const_z).cuda()

    for epoch in range(20):
        train()

        # Progress.
        gen_out = gen(const_z)
        gen_out = gen_out.view(-1, 1, 28, 28)
        save_image(gen_out.data, 'img/generated_at_epoch_{}.png'.format(epoch))

    # Save results.
    torch.save(gen.state_dict(), 'gen_state_dict.pth')
    torch.save(dis.state_dict(), 'dis_state_dict.pth')