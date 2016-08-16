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
    M = np.zeros((size, size))
    for i in np.arange(size):
        for j in np.arange(size):
            if i % period == 0 and j % period == 0 and i != 0 and j != 0:
                m = makeGaussian(size, fwhm=fwhm, center=[xoffset+i, yoffset+j])
                M = M+m
    return M
    
    
def circlesMatrix(size, N, noise):
    """ Makes a matrix with many circles of random centres, it is the image
    to evaluate resolution"""
    
    M = np.zeros((size, size))
    for i in np.arange(N): 
        xx, yy = np.mgrid[:size, :size]
        x0 = np.int(size*np.random.rand())
        y0 = np.int(size*np.random.rand())
        circle = (xx - x0) ** 2 + (yy - y0) ** 2
        donut = (circle < int(size/10+size/500)**2) & (circle > int(size/10-size/500)**2)
        donut = np.array(donut, dtype='int')
        M = M+donut
    return 1000*M

def addNoise(gtruth, offset, nfac):
        imsize = np.shape(gtruth)[0]
        offsetimg = np.zeros([imsize, imsize]) + offset + 4*np.random.randn(imsize, imsize)
        poissonian = np.random.poisson(nfac*gtruth, [imsize, imsize])
        out = offsetimg + poissonian
        
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


def rawStack(image, period, fwhm, step_size):
    """ Generates simulated parallelized RESOLFT raw data stack """
    imageSize = np.shape(image)[0]
    pattern = periodicMatrix(imageSize, period, fwhm, 0, 0)
    stack = []
    
    # scan is done first right, then down
    for i in np.arange(period/step_size):
        for j in np.arange(period/step_size):
            dx = i*step_size
            dy = j*step_size
            frame = pattern*scipy.ndimage.interpolation.shift(image, [dx, dy])
            frame = ndi.gaussian_filter(frame, 5)
            stack.append(frame)
    return stack

size = 400
offset = 110
noise = 0.5

circles = circlesMatrix(size, 56, noise)
circles = np.array(circles, dtype='uint16')
noisedCirc = addNoise(circles, offset, noise)
noisedCirc = np.array(noisedCirc, dtype='uint16')
tifffile.imsave('circles_2period.tif', circles)

darkframe = addNoise(np.ones((size, size)), offset, noise)
darkframe = np.array(darkframe, dtype='uint16')
tifffile.imsave('darkframe.tif', darkframe)

rawstack = rawStack(circles, 16, 8, 0.5)
rawstack = np.array(addNoiseStack(rawstack, offset, noise),dtype='uint16')
tifffile.imsave('rawstack1_p.tif', rawstack)



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
