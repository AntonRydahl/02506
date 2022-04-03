#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:14:57 2020

@author: vand
"""

# optional exercise 1.1.5 

import skimage.io
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

I = skimage.io.imread('../../../../Data/week1/dental/slice100.png')


a = 180 # number of angles for unfolding
angles = np.arange(a)*2*np.pi/a # angular coordinate

center = (np.array(I.shape)-1)/2
r = int(min(I.shape)/2)
radii = np.arange(r) + 1 #radial coordinate for unwrapping

X = center[0] + np.outer(radii,np.cos(angles))
Y = center[1] + np.outer(radii,np.sin(angles))

F = scipy.interpolate.interp2d(np.arange(I.shape[0]), np.arange(I.shape[1]), I)
U = np.array([F(p[0],p[1]) for p in np.c_[Y.ravel(),X.ravel()]])
U = U.reshape((r,a)).astype(np.uint8)

fig, ax = plt.subplots(1,2)
ax[0].imshow(I, cmap='gray')
ax[1].imshow(U, cmap='gray')
