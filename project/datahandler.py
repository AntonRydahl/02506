# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:28:48 2022

@author: lisbe
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

import torch
import skimage.io

import random

import torchvision.transforms as transforms

# Make dataset class.
class DataHandler(torch.utils.data.Dataset):
    '''  Dataset which loads all images for training or testing'''
    def __init__(self, data_dir, im_id, subset, margin_size=20,is_ubuntu=False,
        img_trans= transforms.Compose([transforms.ToTensor()]),
        lab_trans= transforms.Compose([transforms.ToTensor()]),
        seed = 12345):
        self.images = []
        self.image_patches=[]
        self.labels = []
        self.labels_patches=[]
        self.img_trans = img_trans
        self.lab_trans = lab_trans

        # Making sure that we use the same seed every time
        self.seed = seed
        np.random.seed(seed)

        for idx in im_id:
            # Making sure that images can be loaded both on Windows and Ubuntu
            if is_ubuntu:
                self.image_path = f'{data_dir}/{subset}_images/{subset}_{idx:02d}.png'
                self.label_path = f'{data_dir}/{subset}_labels/labels_{idx:02d}.png'
            else:
                self.image_path = f'{data_dir}\{subset}_images\{subset}_{idx:02d}.png'
                self.label_path = f'{data_dir}\{subset}_labels\labels_{idx:02d}.png'

            im = 1.0-torch.tensor(skimage.io.imread(self.image_path),dtype=torch.float32)/255.0
            self.images.append(im)
            label_im = 1.0-torch.tensor(skimage.io.imread(self.label_path),dtype=torch.float32)/255.0
            self.labels.append(label_im)

        for i in range(len(self.images)):
                # patches of image
            patches_image=self.images[i].unfold(0, 128, 128).unfold(1, 128, 128)
            patches_image=patches_image.flatten(0,1)
            
            # patches of label
            patches_label=self.labels[i].unfold(0, 128, 128).unfold(1, 128, 128)
            patches_label=patches_label.flatten(0,1)
            
            for j in range(len(patches_label)):
                self.image_patches.append(np.array(patches_image[j][None,:][0],dtype=np.float32))
                self.labels_patches.append(np.array(patches_label[j][None,:][0],dtype=np.float32))

    def __getitem__(self, idx):
        # Generating new seed from previous seed
        self.seed = np.random.randint(0,high=4294967296)
        torch.manual_seed(self.seed)
        img = self.img_trans(self.image_patches[idx])
        torch.manual_seed(self.seed)
        lab = self.lab_trans(self.labels_patches[idx])
        return img, lab
    
    def __len__(self):
        return len(self.image_patches)
    