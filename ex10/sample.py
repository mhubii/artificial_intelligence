import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from model import VAE

# Load model.
vae = VAE(28*28)
vae.load_state_dict(torch.load('vae_10_epochs.pth'))

# Sample.
sample = torch.randn(4, 64)
sample = Variable(sample)

# Decode.
out = vae.forward_decoder(sample)

out = out.data.numpy()
out = out.reshape([4, 28, 28])

plt.figure()
plt.suptitle('Randomly Generated Samples')

for i in range(4):
    plt.subplot('22{}'.format(i))
    plt.axis('off')
    plt.imshow(out[i])

plt.savefig('img/randomly_generated_samples.png')
