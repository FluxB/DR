

from matplotlib import pyplot as plt
from skimage import data
from scipy import ndimage
from skimage import feature
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage import io
from skimage import exposure
from skimage import transform
from skimage import img_as_ubyte
import numpy as np
import os


def check_blob_intensity(y,x,r, im):
    Delta=int(r/2)
    if abs(y-im.shape[0])<=Delta or abs(x-im.shape[1])<=Delta :
        Delta=0
    
    check=(4*im[y,x]+im[y+Delta,x]+im[y-Delta,x]+im[y,x+Delta]+im[y,x-Delta])//8
    return check

def check_edge_overlapp(y,x,r,edge):
    if abs(y-im.shape[0])<=r or abs(x-im.shape[1])<=r :
        r=0
    overlapp=np.zeros(edge.shape, dtype=bool)
    rr, cc = circle(y, x, r)
    overlapp[rr,cc]=edge[rr,cc]
    return np.sum(overlapp)


image=io.imread('dropbox/retinopathy/sample/13_left.jpeg')
img_resc=transform.rescale(rgb2gray(image),0.25)
img_gray = img_as_ubyte(exposure.equalize_adapthist(img_resc,clip_limit=0.3))
#plt.imshow(img_gray, cmap = plt.get_cmap('gray'))
#io.imsave('dropbox/retinopathy/bw.jpeg',img_gray)


#Invert White<->Black
image_gray=255-img_gray

image_gray_rgb=gray2rgb(img_gray)

#Make blurry image for edge detection
im = ndimage.gaussian_filter(img_gray, 4)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im, sigma=0.1)
edges2 = feature.canny(im, sigma=3)

# Image with edges included
image_gray_rgb[edges1]=(255,0,0)



fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))

ax0.imshow(img_gray, cmap=plt.cm.gray)
ax0.axis('off')
ax0.set_title('Gray Scale')

ax1.imshow(edges1, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Edge Detection')

ax2.set_title('Blob Detection')
ax2.imshow(gray2rgb(img_gray), interpolation='nearest')
ax2.axis('off')

ax3.set_title('Combined')
ax3.imshow(image_gray_rgb, interpolation='nearest')
ax3.axis('off')


#Blob Detection

blobs_1 = blob_log(image_gray,min_sigma=3, max_sigma=10, num_sigma=8, threshold=.15)
# Compute radii in the 3rd column.
blobs_1[:, 2] = blobs_1[:, 2] * sqrt(2)

blobs_2 = blob_doh(img_gray,min_sigma=5, max_sigma=25,num_sigma=15, threshold=.005)

blobs_small=np.zeros((blobs_1.shape[0],blobs_1.shape[1]+2), dtype=np.int)
blobs_small[:,0:3]=blobs_1[:,:]
blobs_small[:,3]=0
blobs_small[:,4]=1


blobs_large=np.zeros((blobs_2.shape[0],blobs_2.shape[1]+2), dtype=np.int)
blobs_large[:,0:3]=blobs_2[:,:]
blobs_large[:,3]=1
blobs_large[:,4]=2

colors=['yellow','orange']

blobs_list = [blobs_small,blobs_large]



for blobs in blobs_list:
    for i in range(blobs.shape[0]):
        y, x, r, c_nb, tag = blobs[i,:]
        c_tag=colors[c_nb]
        intens=check_blob_intensity(y,x,r, img_gray)
        overl=check_edge_overlapp(y,x,r*1.4,edges1)
        c = plt.Circle((x, y), r, color=c_tag, linewidth=2, fill=False)
        #if overl >0 :
        #        c='blue'
        ax2.add_patch(c)
        if (intens <50 and r>10) or (intens<100 and r<11 and overl<2):
            c = plt.Circle((x, y), r, color=c_tag, linewidth=2, fill=False)
            ax3.add_patch(c)

fig.subplots_adjust(wspace=0.02, hspace=0.08, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)
plt.show()
