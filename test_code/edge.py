"""
===================
Canny edge detector
===================

The Canny filter is a multi-stage edge detector. It uses a filter based on the
derivative of a Gaussian in order to compute the intensity of the gradients.The
Gaussian reduces the effect of noise present in the image. Then, potential
edges are thinned down to 1-pixel curves by removing non-maximum pixels of the
gradient magnitude. Finally, edge pixels are kept or removed using hysteresis
thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the
noisier the image, the greater the width), and the low and high threshold for
the hysteresis thresholding.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage import io
from skimage import exposure
from skimage import transform
from skimage import img_as_ubyte
import os


from skimage import feature

image=io.imread('dropbox/retinopathy/sample/16_left.jpeg')
img_resc=transform.rescale(rgb2gray(image),0.2)
img_gray = img_as_ubyte(exposure.equalize_adapthist(img_resc,clip_limit=0.2))
#plt.imshow(img_gray, cmap = plt.get_cmap('gray'))
#io.imsave('dropbox/retinopathy/bw.jpeg',img_gray)
print img_gray.shape
print img_gray[250,250]
image_gray=img_gray
image_gray_rgb=gray2rgb(img_gray)
print np.max(image_gray)
im=image_gray
# Generate noisy image of a square
#im = np.zeros((128, 128))
#im[32:-32, 32:-32] = 1

#im = ndimage.rotate(im, 15, mode='constant')
im = ndimage.gaussian_filter(im, 4)
im += (0 * np.random.random(im.shape)).astype(int)
too_high=im>255
im[too_high]=255
print np.max(im)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im, sigma=0.15)
edges2 = feature.canny(im, sigma=3)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax1.imshow(im, cmap=plt.cm.gray)
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

