#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:47:29 2020

@author: abda
"""


import numpy as np
#%%

def make_data(example_nr, n = 200, noise = 1):
# Generate data for training a simple neural network.
# 
# def make_data(example_nr, n = 200, noise = 1):
# ...
#     return X, T, x, dim
# 
# Input:
#   example_nr - a number 1 - 3 for each example
#   n - number of points in each data set
#   noise - a number to increase or decrease the noise level (if changed, 
#       choose between 0.5 and 2)
# Output:
#   X - 2n x 2 array of points (there are n points in each class)
#   T - 2n x 2 target values
#   x - regular sampled points on the area covered by the points that will
#       be used for testing the neural network
#   dim - dimensionality of the area covered by the points
# 
# Authors: Vedrana Andersen Dahl and Anders Bjorholm Dahl - 25/3-2020
#   vand@dtu.dk, abda@dtu.dk
# 

    if ( n % 2 == 1 ):
        n += 1
    dim = np.array([100, 100])
    QX, QY = np.meshgrid(range(0,dim[0]), range(0,dim[1]))
    x = np.c_[np.ravel(QX), np.ravel(QY)]
    K = np.array([n,n])
    T = np.r_[np.ones((n,1))*np.array([1,0]), np.ones((n,1))*np.array([0,1])]
    if example_nr == 1 :
        X = np.r_[noise*10*np.random.randn(K[0],2) + np.array([30,30]),
                  noise*10*np.random.randn(K[1],2) + np.array([70,70])]
    elif example_nr == 2 :
        rand_ang = np.random.rand(K[0])*2*np.pi
        X = np.r_[noise*5*np.random.randn(K[0],2) + 30*np.array([np.cos(rand_ang), np.sin(rand_ang)]).T, 
                  noise*5*np.random.randn(K[1],2)] + dim/2
    elif example_nr == 3 :
        X = np.r_[noise*10*np.random.randn(int(K[0]/2),2) + np.array([30,30]), 
                  noise*10*np.random.randn(int(K[0]/2),2) + np.array([70,70]),
                  noise*10*np.random.randn(int(K[1]/2),2) + np.array([30,70]),
                  noise*10*np.random.randn(int(K[1]/2),2) + np.array([70,30])]
    else:
        X = np.zeros((K[0] + K[1],2))
        print('No data returned - example_nr must be 1, 2, or 3')
    return X, T, x, dim

# Test of the data generation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    n = 1000
    example_nr = 2
    noise = 1.2
    
    X, T, x, dim = make_data(example_nr, n, noise)
    fig, ax = plt.subplots(1,1)
    ax.scatter(X[0:n,0],X[0:n,1],c = 'red', alpha = 0.3, s = 15)
    ax.scatter(X[n:2*n,0],X[n:2*n,1],c = 'green', alpha = 0.3, s = 15)
    ax.set_aspect('equal', 'box')
    plt.title('training')
    fig.show
    
    
    
    #%% Before training, you should make data have zero mean and std of 1
    
    c = X.mean(axis = 0)
    std = X.std(axis = 0)
    x_c = (x - c)/std
    X_c = (X - c)/std
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(X_c[0:n,0],X_c[0:n,1],c = 'red', alpha = 0.3, s = 15)
    ax.scatter(X_c[n:2*n,0],X_c[n:2*n,1],c = 'green', alpha = 0.3, s = 15)
    ax.set_aspect('equal', 'box')
    plt.title('Zero mean training')
    fig.show







