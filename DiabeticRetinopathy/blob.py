# -*- coding: utf-8 -*-



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
from skimage.data import camera
from skimage.filter import threshold_otsu
from skimage.filter import threshold_adaptive
from skimage.filter import threshold_yen
from skimage.restoration import denoise_bilateral
from skimage.transform import rotate
from skimage.restoration import denoise_tv_bregman
import math
import numpy as np
import os


def matched_gaussian_filter_oneDirection(image, sigma, width, L):
    range_x = image.shape[0]
    range_y = image.shape[1]    
    
    #Generate the template used for matching. This needs to be done only once, only the calibration level has to be adjusted for each pixel
    template = np.zeros((2*width + 1, range_y))
    
    f_sigma = float(sigma)
    
    for i in range(-width, width + 1, 1):
        template[i + width, :] = -1/math.sqrt(2*math.pi*f_sigma*f_sigma)*math.exp(-float(i*i)/(2*f_sigma*f_sigma))
    
    
    convoluted_image = np.zeros((range_x, range_y))
    calibration_level = np.zeros((range_x, range_y))
    convolve_helper = np.ones(L)
    
    for x in range(width, range_x - width):
        convoluted_image[x, :] = np.sum(template * image[(x - width):(x + width + 1), :], axis=0)
        calibration_level[x, :] = np.sum(image[(x - width):(x + width + 1), :], axis=0)
        convoluted_image[x, :] = np.convolve(convoluted_image[x, :]/calibration_level[x, :], convolve_helper, mode = 'same')
        #convoluted_image[x, :] = np.convolve(convoluted_image[x, :], convolve_helper, mode = 'same')
        convoluted_image[x, :] = np.convolve(convoluted_image[x, :], convolve_helper, mode = 'same')
            
    return convoluted_image
    
def matched_gaussian_filter(image, sigma, width, L):
    #plt.imshow(image)
    #plt.show()
    angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    img_shape = image.shape    
    
    final_image = np.zeros(img_shape)
    
    for angle in angles: 
        rotated_image = rotate(image, angle, resize = True)
        filtered_image = matched_gaussian_filter_oneDirection(rotated_image, sigma, width, L)
        filtered_image = rotate(filtered_image, -angle, resize=True)
        filtered_img_shape = filtered_image.shape
        filtered_image = filtered_image[(filtered_img_shape[0] - img_shape[0])/2:(filtered_img_shape[0] + img_shape[0])/2, (filtered_img_shape[1] - img_shape[1])/2:(filtered_img_shape[1] + img_shape[1])/2]
        
        if angle == 0: 
            final_image = filtered_image
        else:
            mask = filtered_image > final_image
            final_image = mask * filtered_image + (1-mask) * final_image
    
    #thresh = threshold_li(final_image)
    #binary = final_image > thresh
    #edges = feature.canny(final_image, sigma=1)
    #final_image = denoise_tv_bregman(final_image,0.01)
    plt.imshow(final_image)
    plt.show()
        
    

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


image=io.imread('/Users/Felix/Dropbox/Diabetic Retinopathy/sample/17_left.jpeg')
image[:,:,0] = 0
image[:,:,2] = 0
img_resc=transform.rescale(rgb2gray(image),0.25)
img_gray = img_as_ubyte(exposure.equalize_adapthist(img_resc,clip_limit=0.3))
#plt.imshow(img_gray, cmap = plt.get_cmap('gray'))
#io.imsave('dropbox/retinopathy/bw.jpeg',img_gray)

#Invert White<->Black
image_gray=255-img_gray

image_gray_rgb=gray2rgb(img_gray)

#thresh =threshold_otsu(image_gray_rgb)
#plt.imshow(binary, interpolation='nearest')
#binary = image_gray_rgb > thresh

#Make blurry image for edge detection
im = ndimage.gaussian_filter(img_gray, 4)

matched_gaussian_filter(im, 1, 10, 4)

quit()

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im, sigma=0.1) #, low_threshold = 18, high_threshold = 20)
edges2 = feature.canny(im, sigma=3)

# Image with edges included
image_gray_rgb[edges1]=(255,0,0)

#plt.plot(img_gray[585:605, 650-55])
#plt.plot(img_gray[55, 585:605])
#plt.show()

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
