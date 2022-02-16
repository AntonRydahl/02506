"""
Helping functions for extracting features from images

Vedrana Andersen Dahl (vand@dtu.dk) 
Anders Bjorholm Dahl (abda@dtu.dk)
"""

import numpy as np
import scipy.ndimage

def get_gauss_feat_im(im, sigma=1, normalize=True):
    """Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r,c).
        sigma: standard deviation for Gaussian derivatives.
        normalize: flag indicating normalization of features.
    Returns:
        imfeat: 3D array of size (r,c,15) with a 15-dimentional feature
             vector for every pixel in the image.
    Author: vand@dtu.dk, 2020
    """
      
    r,c = im.shape
    imfeat = np.zeros((r,c,15))
    imfeat[:,:,0] = scipy.ndimage.gaussian_filter(im,sigma,order=0)
    imfeat[:,:,1] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,1])
    imfeat[:,:,2] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,0])
    imfeat[:,:,3] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,2])
    imfeat[:,:,4] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,1])
    imfeat[:,:,5] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,0])
    imfeat[:,:,6] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,3])
    imfeat[:,:,7] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,2])
    imfeat[:,:,8] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,1])
    imfeat[:,:,9] = scipy.ndimage.gaussian_filter(im,sigma,order=[3,0])
    imfeat[:,:,10] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,4])
    imfeat[:,:,11] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,3])
    imfeat[:,:,12] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,2])
    imfeat[:,:,13] = scipy.ndimage.gaussian_filter(im,sigma,order=[3,1])
    imfeat[:,:,14] = scipy.ndimage.gaussian_filter(im,sigma,order=[4,0])

    if normalize:
        imfeat -= np.mean(imfeat, axis=(0,1))
        imfeat /= np.std(imfeat, axis=(0,1))
    
    return imfeat

def get_gauss_feat_multi(im, sigma = [1,2,4], normalize = True):
    '''Multi-scale Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r,c).
        sigma: list of standard deviations for Gaussian derivatives.
        normalize: flag indicating normalization of features.
    Returns:
        imfeat: a a 3D array of size (r*c,n_scale,15) with n_scale features in each pixels, and
             n_scale is length of sigma. Each pixel contains a feature vector and feature
             image is size (r,c,15*n_scale).
    Author: abda@dtu.dk, 2021

    '''
    imfeats = []
    for i in range(0,len(sigma)):
        feat = get_gauss_feat_im(im, sigma[i], normalize)
        imfeats.append(feat.reshape(-1,feat.shape[2]))
    
    imfeats = np.asarray(imfeats).transpose(1,0,2)
    return imfeats


def im2col(im, patch_size=[3,3], stepsize=1):
    """Rearrange image patches into columns
    Arguments:
        image: a 2D image, shape (r,c).
        patch size: size of extracted paches.
        stepsize: patch step size.
    Returns:
        patches: a 2D array which in every column has a patch associated 
            with one image pixel. For stepsize 1, number of returned column 
            is (r-patch_size[0]+1)*(c-patch_size[0]+1) due to bounary. The 
            length of columns is pathc_size[0]*patch_size[1].
    """
    
    r,c = im.shape
    s0, s1 = im.strides    
    nrows =r-patch_size[0]+1
    ncols = c-patch_size[1]+1
    shp = patch_size[0],patch_size[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(im, shape=shp, strides=strd)
    return out_view.reshape(patch_size[0]*patch_size[1],-1)[:,::stepsize]


def ndim2col(im, block_size=[3,3], stepsize=1):
    """Rearrange image blocks into columns for N-D image (e.g. RGB image)"""""
    if(im.ndim == 2):
        return im2col(im, block_size, stepsize)
    else:
        r,c,l = im.shape
        patches = np.zeros((l*block_size[0]*block_size[1],
                            (r-block_size[0]+1)*(c-block_size[1]+1)))
        for i in range(l):
            patches[i*block_size[0]*block_size[1]:(i+1)*block_size[0]*block_size[1],
                    :] = im2col(im[:,:,i],block_size,stepsize)
        return patches

#%%
import skimage.io
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    filename = './3labels/training_image.png'
    I = skimage.io.imread(filename)
    I = I.astype(np.float)
    
    
    # features based on gaussian derivatives
    sigma = 1;
    gf = get_gauss_feat_im(I, sigma)
    
    fig,ax = plt.subplots(3,5)
    for j in range(5):
        for i in range(3):
            ax[i][j].imshow(gf[:,:,5*i+j], cmap='jet')
            print(np.std(gf[:,:,5*i+j]))
            ax[i][j].set_title(f'layer {5*i+j}')
            
            
    # features based on image patches
    I = I[300:320,400:420] # smaller image such that we can see the difference in layers
    pf = im2col(I,[3,3])
    pf = pf.reshape((9,I.shape[0]-2,I.shape[1]-2))
            
    fig,ax = plt.subplots(3,3)
    for j in range(3):
        for i in range(3):
            ax[i][j].imshow(pf[3*i+j], cmap='jet')
            ax[i][j].set_title(f'layer {3*i+j}')
   
    
