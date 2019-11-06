import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature
from skimage.filters import roberts, sobel

# READ DATA ####################################################################

# About 1 minute to load
dat = pd.read_csv('../data/training.csv', sep='\s|,', engine='python',
                  header=1, index_col=False).values

faces = dat[:, 30:]

im = faces[0,]
im = np.reshape(im, (96, 96))

# CANNY EDGE DETECTION #########################################################

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax1.imshow(im, cmap=plt.cm.Greys_r)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()

# ROBERTS AND SOBEL EDGE DETECTION #############################################

edge_roberts = roberts(im)
edge_sobel = sobel(im)

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)

ax0.imshow(im, cmap=plt.cm.gray)
ax0.set_title('Original Image')
ax0.axis('off')

ax1.imshow(edge_roberts, cmap=plt.cm.gray)
ax1.set_title('Roberts Edge Detection')
ax1.axis('off')

ax2.imshow(edge_sobel, cmap=plt.cm.gray)
ax2.set_title('Sobel Edge Detection')
ax2.axis('off')

plt.tight_layout()

plt.show()
