# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:36:06 2016

@author: Luciano
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
import scipy.signal
import tifffile
from skimage import io


def importData(cameraFrameFile, darkFrameFile):
    
    img = Image.open(darkFrameFile)
    background = np.array(img) 
    
    data = io.imread(cameraFrameFile)
    data = np.array(data)
    
    return data, background
    
    
def getOffPattern(data, expectedValue):

    """data are the camera frames, expected_value is the initial guess on the
    period of the pattern in pixels

     output is a vector with the following order:
         - period in x-direction [pixels] (px)
         - offset in x-direction [pixels] (x0)
         - period in y-direction [pixels] (py)
         - offset in y-direction [pixels] (y0)
     and the function for recreating the off switching pattern would be:
     sin(pi * (x - x0) / px).^2 + sin(pi * (y - y0) / py).^2"""

    pattern = 0

    return pattern   

def objectDistances     
    
  
def signalReconstruction(data, pattern, objp, shiftp):  
    
    """ Given the pattern period and offset as well as the output pixel length
    and the scanning pixel length it constructs the central and peripheral
    signal frames. Pattern is the periods and offsets of the on-switched 
    regions"""

    # data parameters
    dx = np.size(data)[1]
    dy = np.size(data)[2]
    nframes = np.size(data)[0]
    nsteps = np.sqrt(nframes)

    # decode the pattern
    fx = pattern[0]
    x0 = pattern[1]
    fy = pattern[2]
    y0 = pattern[3]

    # object positions in image so that they are always in the scanned regions
    [xi, yi] = object_positions([fx, dx], [1, dy - fy], objp)

    # central loop: interpolate camera frame on shifting grids (scanning)
    # and extract central and peripheral signals
    central_signal = 0
    central_signal_weights = 0
    peripheral_signal = 0
    peripheral_signal_weights = 0

    # loop (attention, the scanning direction of our microscope is hardcoded,
    # first down, then right)
    for kx in np.arange(nsteps):
        shift_x = -kx * shiftp
    
        for ky in np.arange(nsteps):
            shift_y = ky * shiftp;
            
            # get frame number and frame
            kf = ky + 1 + nsteps * kx
            frame = data(:, :, kf)
            
            # adjust positions for this frame
            xj = xi + shift_x
            yj = yi + shift_y
            
            # interpolation
            est = interpn(frame, xj, yj, 'nearest');
            
            # result will be isnan for outside interpolation (should not happen)
            est(isnan(est)) = 0;
            est = max(est, 0); % no negative values (should only happen rarely)
            
            # compute distance to the center (minima of off switching pattern)
            [t2max, ~] = objectDistances(xj, yj, fx, x0, fy, y0)
        
            # compute weights (we add up currently 50nm around each position),
            # feel free to change this value for tradeoff of SNR and resolution
            W = 0.05 / 0.0975;
            wmax = power(2., -t2max / (W / 2)^2)
            
            # add up with weights
            central_signal = central_signal + wmax .* est
            central_signal_weights = central_signal_weights + wmax
            
            # subtraction of surrounding minima
            cx = round(fx / 2 / objp);
            cy = round(fy / 2 / objp);
            
            # left upper
            shifted = circshift(est, [-cx, -cy]);
            peripheral_signal = peripheral_signal + wmax .* shifted
            peripheral_signal_weights = peripheral_signal_weights + wmax
            
            # another
            shifted = circshift(est, [cx, -cy]);
            peripheral_signal = peripheral_signal + wmax .* shifted;
            peripheral_signal_weights = peripheral_signal_weights + wmax
            
            # another
            shifted = circshift(est, [-cx, cy]);
            peripheral_signal = peripheral_signal + wmax .* shifted;
            peripheral_signal_weights = peripheral_signal_weights + wmax
            
            # another
            shifted = circshift(est, [cx, cy]);
            peripheral_signal = peripheral_signal + wmax .* shifted;
            peripheral_signal_weights = peripheral_signal_weights + wmax
            
            
            # normalize by weights
            central_signal = central_signal ./ central_signal_weights;
            peripheral_signal = peripheral_signal ./ peripheral_signal_weights


    return central_signal, peripheral_signal
    
    
    
data, background = importData(r'/Users/Luciano/Documents/LabNanofisica/rawstack.tif',
                              r'/Users/Luciano/Documents/LabNanofisica/darkframe.tif')

## some physical parameters of the setup

camera_pixel_length = 0.0975   # camera pixel length [µm] in sample space
scanning_period = 0.322      # scanning period [µm] in sample space
number_scanning_steps = 12     # number of scanning steps in one direction
    # total number of camera frames is (number_scanning_steps)^2
pixel_length = 0.02            # pixel length [µm] of interpolated and combined frames

# derived parameters
shift_per_step = scanning_period / number_scanning_steps / camera_pixel_length;
    # shift per scanning step [camera pixels]
pixel_length_per_camera = pixel_length / camera_pixel_length;
    # length of pixel of combined frames in camera pixels