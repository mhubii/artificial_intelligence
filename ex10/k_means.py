import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.misc import imread


# Load image.
img = imread('grass2.jpg')
img = np.reshape(img, [img.shape[0]*img.shape[1], img.shape[2]])

# Apply clustering.
n_clusters = 3

k_means = KMeans(n_clusters=n_clusters).fit(img)
labels = k_means.labels_

# Plot results.
r = img[:, 0]
g = img[:, 1]
b = img[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(r, g, b, c=labels.astype(np.float))
ax.set_title('k-Means Clustering using {} Clusters'.format(n_clusters))
ax.set_xlabel('r')
ax.set_ylabel('g')
ax.set_zlabel('b')
plt.savefig('img/k_means_{}_clusters.png'.format(n_clusters))
