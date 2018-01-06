from exercise_09_starter import Autoencoder
import numpy as np
import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# Training settings.
parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# Load data set.
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True)

# Build autoencoder.
input_shape = (28, 28)
model = Autoencoder(input_shape)

# Optimizer and loss.
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()


# Define train and test.
def train(epoch):
    model.train()
    loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    return loss.data[0]


def test():
    model.eval()
    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, data)

        # Return loss, data and output for the last sample.
        if batch_idx % args.test_batch_size == batch_idx:
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}\n'.format(float(test_loss)))
            return test_loss.data[0], \
                   data[0].data.numpy().reshape(input_shape), \
                   output[0].data.numpy().reshape(input_shape)


# Train.
lowest_loss = float('inf')

fig_encoder = plt.figure()
fig_encoder.suptitle('Autoencoder')

train_loss_log = np.empty(args.epochs)
test_loss_log = np.empty(int(args.epochs/5))

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    test_loss, input, output = test()

    train_loss_log[epoch - 1] = train_loss

    if epoch % 5 == 0:
        test_loss_log[int(epoch/5 - 1)] = test_loss

    if epoch % 10 == 0:
        plt.subplot(121)
        plt.title('Original Image')
        plt.imshow(input)

        plt.subplot(122)
        plt.title('Decoded at Epoch {}'.format(epoch))
        plt.imshow(output)

        plt.savefig('img/autoencoder_epoch_{}.png'.format(epoch))

    if float(test_loss) < lowest_loss:
        print('New lowest loss achieved: {:.2f}%\n'.format(float(test_loss)))
        lowest_loss = test_loss
        torch.save(model.state_dict(), 'best_model_autoencoder.pth')

# Plot loss.
fig_loss = plt.figure()
fig_loss.suptitle('Loss Progress')

epochs = np.arange(1, args.epochs + 1)
plt.subplot(121)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.plot(epochs, train_loss_log, 'b')

epochs = np.linspace(1, args.epochs + 1, 10)
plt.subplot(122)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.plot(epochs, test_loss_log, 'r')

plt.subplots_adjust(wspace=1)

plt.savefig('img/loss_progress.png')

