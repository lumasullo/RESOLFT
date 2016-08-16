# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 21:41:49 2016

@author: Luciano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
import scipy.signal
import tifffile
import time

upsamp_for_GT = 10

def nm2px(nm):
    px_size = 66
    return upsamp_for_GT * (nm / px_size)

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def periodicMatrix(size, period, fwhm, xoffset, yoffset):
    """ Makes a matrix with a periodic distribution of gaussians, simulates
    parallelized RESOLFT on-state PSF
    """
    nr_of_gauss = np.floor(size / period) # Number of gaussians per dimension  
    gauss_pos = period + np.arange(nr_of_gauss) * period
    
    M = np.zeros((size, size))
    for i in gauss_pos:
        for j in gauss_pos:
            print('Making Gaussian')
            x = np.round(xoffset + i)
            y = np.round(yoffset + j)
            m = makeGaussian(size, fwhm=fwhm, center=[x, y])
            M = M+m
    return M
    
    
def circlesMatrix(size, N, noise):
    """ Makes a matrix with many circles of random centres, it is the image
    to evaluate resolution"""
    
    size = upsamp_for_GT*size    
    
    M = np.zeros((size, size))
    for i in np.arange(N): 
        xx, yy = np.mgrid[:size, :size]
        x0 = np.int(size*np.random.rand())
        y0 = np.int(size*np.random.rand())
        circle = (xx - x0) ** 2 + (yy - y0) ** 2
        donut = (circle < int(size/10+size/500)**2) & (circle > int(size/10-size/500)**2)
        donut = np.array(donut, dtype='int')
        M = M+donut
    return 10000*M

def addNoise(gtruth, offset, nfac):
        imsize = np.shape(gtruth)[0]
        offsetimg = np.zeros([imsize, imsize]) + offset + 4*np.random.randn(imsize, imsize)
        noise = np.random.poisson(gtruth, [imsize, imsize]) + nfac*np.random.randn(imsize, imsize)
        out = np.abs(offsetimg + noise)
        
        return out
        
def addNoiseStack(stack, offset, nfac):
    frames = np.shape(stack)[0]
    noised = []
    for i in reversed(np.arange(frames)):
        frame = stack[i]
        noised.append(addNoise(frame, 110, nfac))
        
    return noised
        
#def rawStack(image, period, fwhm):
#    """ Generates simulated parallelized RESOLFT raw data stack """
#    imageSize = np.shape(image)[0]
#    stack = []
#    for i in np.arange(32):
#        for j in np.arange(32):
#            frame = periodicMatrix(imageSize, period, fwhm, i, j)*image
#            frame = (10**4)*ndi.gaussian_filter(frame, 5)
#            stack.append(frame)
#    return stack


def rawStack(image, period, fwhm, steps_per_line):
    """ Generates simulated parallelized RESOLFT raw data stack """
    period_px = nm2px(period)
    fwhm_px = nm2px(fwhm)
    step_size_px = nm2px(period/steps_per_line)
    GT_size = np.shape(image)[0]
    OutSize = GT_size / upsamp_for_GT
    size_ratio = 1/upsamp_for_GT
    print('Creating pattern')
    print(GT_size, period_px, fwhm_px)
    pattern = periodicMatrix(GT_size, period_px, fwhm_px, 0, 0)
    stack = []
    
    # scan is done first right, then down
    for i in np.arange(period_px/step_size_px):
        for j in np.arange(period_px/step_size_px):
            print('Line: ', i)
            dx = i*step_size_px
            dy = j*step_size_px
            print(dx, dy)
            shifted = scipy.ndimage.interpolation.shift(image, [dx, dy])
            frame = pattern*shifted
            frame = ndi.gaussian_filter(frame, 45)
            plt.imshow(frame)
            stack.append(scipy.misc.imresize(frame, size_ratio))
    return stack

size = 200
offset = 110
noise = 10
activation = 200
period = 1000
steps_per_line = 10
noise = False
make_simulated_data = False
print('Making GT')
circles = circlesMatrix(size, 56, noise)
circles = np.array(circles, dtype='uint16')
print('Noising GT')
noisedCirc = addNoise(circles, offset, noise)
noisedCirc = np.array(noisedCirc, dtype='uint16')
tifffile.imsave('circles_large.tif', circles)
print('Creating rawdata')
rawstack = rawStack(circles, period, activation, steps_per_line)

if not noise:
    rawstack = np.array(rawstack,dtype='uint16')
else:
    rawstack = np.array(addNoiseStack(rawstack, offset, noise),dtype='uint16')
    
filename = 'rawstack_.tif'
tifffile.imsave(filename, rawstack)


noise_levels = [0, 10, 100, 500]
offsets = [100, 200, 400, 1000]
activation_sizes = [1.5, 3, 8, 16]

if make_simulated_data == True:
    for n in noise_levels:
        for o in offsets:
            for a in activation_sizes:
                darkframe = addNoise(np.ones((size, size)), o, n)
                darkframe = np.array(darkframe, dtype='uint16')
                filename = 'dark_frame_noise_'+str(n)+'_offset_'+str(o)+'_activation_'+str(a)+'.tif'
                tifffile.imsave(filename, darkframe)
                
                rawstack = rawStack(circles, period, a, step_size)
                rawstack = np.array(addNoiseStack(rawstack, o, n),dtype='uint16')
                filename = 'rawstack_'+str(n)+'_offset_'+str(o)+'_activation_'+str(a)+'.tif'
                tifffile.imsave(filename, rawstack)
            







#rawdata = ndi.gaussian_filter(circles*activated, 5)
#
#plt.figure()
#plt.imshow(activated,interpolation='none',cmap='hot')
#
#plt.figure()
#plt.imshow(scipy.ndimage.interpolation.shift(circles,[30,0]),interpolation='none',cmap='hot')
#
#plt.figure()
#plt.imshow(rawdata,interpolation='none',cmap='hot')
