from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.autograd import Variable


# Manifold approximation using tSNE
#
# features: (numpy array) N x D feature matrix
# images: (numpy array) N x H x W x 3
# path_save: string (path where you want to save the final image)
def apply_tnse_img(features, images, path_save='./tsne_img.png'):

    tnse = TSNE(n_components=2, init='pca', n_iter=1000, random_state=1254, perplexity=30, metric='euclidean')
    # np.set_printoptions(False)
    vis_data = tnse.fit_transform(features)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    # get max heigth, width
    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])

    # get max, min coords
    x_min, x_max = vis_x.min(), vis_x.max()
    y_min, y_max = vis_y.min(), vis_y.max()

    # Fix the ratios
    res = 700
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    if sx > sy:
        res_x = int(sx / float(sy) * res)
        res_y = res
    else:
        res_x = res
        res_y = int(sy / float(sx) * res)

    # impaint images
    canvas = np.ones((res_x + max_width, res_y + max_height, 3))
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image in zip(vis_x, vis_y, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords) ** 2)
        y_idx = np.argmin((y - y_coords) ** 2)
        try:
            canvas[x_idx:x_idx + w, y_idx:y_idx + h] = image
        except:
            print('Image out of borders.... skip!')

    # plot image
    fig = plt.figure()
    plt.imshow(canvas)

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show(False)
    plt.pause(3)
    fig.savefig(path_save, bbox_inches='tight')


# Reparametrization trick for training VAEs
#
# mu and logvar: output of your encoder
def reparameterize(self, mu, logvar):
    if self.training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


# VAE Loss
#
# recon_x: image reconstructions
# x: images
# mu and logvar: outputs of your encoder
# batch_size: batch_size
# img_size: width, respectively height of you images
# nc: number of image channels
def loss_function(recon_x, x, mu, logvar, batch_size, img_size, nc):
    MSE = F.mse_loss(recon_x, x.view(-1, img_size * img_size * nc))

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalize
    KLD /= batch_size * img_size * img_size * nc

    return MSE + KLD