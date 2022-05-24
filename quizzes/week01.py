#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 23:07:36 2021

@author: vand
"""

import numpy as np
import scipy
import skimage.io
import matplotlib.pyplot as plt

#%% QUESTION 1
I_noisy = skimage.io.imread('../../../../Data/week1/noisy_number.png').astype(np.float)
sigma = 15;
I_smoothed = scipy.ndimage.gaussian_filter(I_noisy, sigma, mode='nearest')

fig, ax = plt.subplots(1,2)
ax[0].imshow(I_noisy)
ax[0].set_title('Noisy image')
ax[1].imshow(I_smoothed)
ax[1].set_title(f'Smoothed with sigma={sigma}')

#%% QUESTION 2
def boundary_length(S):
    L = np.sum(S[1:,:]!=S[:-1,:])+np.sum(S[:,1:]!=S[:,:-1])
    return L

fig, ax = plt.subplots(1,3)
for i in range(3):
    name = f'fuel_cell_{i+1}.tif'
    I = skimage.io.imread('../../../../Data/week1/fuel_cells/' + name)
    L = boundary_length(I)
    ax[i].imshow(I)
    ax[i].set_title(f'{name}\nL = {L}')

#%% QUESTION 3
X_noisy = np.loadtxt('../../../../Data/week1/curves/' + 'dino_noisy.txt')
N = X_noisy.shape[0]

def curve_length(X):
    d = (np.sqrt(((X-np.roll(X, shift=1, axis=0))**2).sum(axis=1))).sum()
    return(d)

a = np.array([-2, 1, 0]) 
D = np.fromfunction(lambda i,j: np.minimum((i-j)%N,(j-i)%N), (N,N), dtype=np.int)
L = a[np.minimum(D,len(a)-1)]

X_solution = np.matmul(0.25*L+np.eye(N),X_noisy)

closed_ind = np.r_[np.arange(N),0] # for easy plotting a closed snake
fig, ax = plt.subplots()
ax.plot(X_noisy[closed_ind,0], X_noisy[closed_ind,1],'r')
ax.plot(X_solution[closed_ind,0], X_solution[closed_ind,1],'b--')
ax.set_title(f'length noisy: {curve_length(X_noisy):.5g}\n'+
             f'lenght smoothed: {curve_length(X_solution):.5g}')
ax.axis('equal')

