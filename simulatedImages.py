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

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

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
    
    
def circlesMatrix(size, N):
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
        
    return M


    
def rawStack(image):
    """ Generates simulated parallelized RESOLFT raw data stack """
    stack = []
    for i in np.arange(16):
        for j in np.arange(16):
            frame = periodicMatrix(500, 16, 2.5,i,j)*circles
            frame = (10**4)*ndi.gaussian_filter(frame, 5)
            stack.append(frame)
    return stack


circles = circlesMatrix(500, 56)
circles = np.array(10*circles, dtype='uint16')
tifffile.imsave('circles.tif', circles)

rawstack = rawStack(circles)
rawstack = np.array(rawstack,dtype='uint16')
tifffile.imsave('rawstack.tif', rawstack)



#rawdata = ndi.gaussian_filter(circles*activated, 5)
#
#plt.figure()
#plt.imshow(activated,interpolation='none',cmap='hot')
#
#plt.figure()
#plt.imshow(circles,interpolation='none',cmap='hot')
#
#plt.figure()
#plt.imshow(rawdata,interpolation='none',cmap='hot')
