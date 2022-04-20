# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:15:23 2022

@author: lisbe
"""
import torch
import torch.nn as nn

# Creating a double convolutional layer with ReLU activation 
def double_conv(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
        nn.ReLU(inplace=True)
        )
# Creating the last convolutional layer withou ReLU
def last_conv(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size),
        nn.Softmax(dim=1)
    )

# Creating the network
class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 8,3)
        self.dconv_down2 = double_conv(8, 16, 3)
        self.dconv_down3 = double_conv(16, 32, 3)
        #self.dconv_down4 = double_conv(32, 64, 3)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.upconv2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dconv_up2 = double_conv(32, 16, 3)
        self.upconv1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dconv_up1 = double_conv(16, 8, 3)
        #self.upconv2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        #self.dconv_up2 = double_conv(16, 8, 3)
        
        #self.dconv_up1 = double_conv(8 + 16, 16, 3)
        
        self.conv_last = last_conv(8,2,1) 
        
        
    def forward(self, x):
        
        # Layer 1 down
        #print('input image')
        #print(x.shape)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        #print('after first layer and max')
        #print(x.shape)
        
        # Layer 2 down
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        #print('after second layer with maxpool')
        #print(x.shape)
        
        # Layer 3 down
        #conv3 = 
        x = self.dconv_down3(x)
        #print('after third layer')
        #print(x.shape)

        # Layer 4 (bottom layer of the U)
        #x = self.dconv_down4(x)
        #print('after bottom layer')
        #print(x.shape)
        
        # Layer 2 up
        
        x=self.upconv2(x)
        #print('after upsample layer 2up')
        #print(x.shape)
        x = torch.cat([x, conv2], dim=1)
        #print('after cat layer 2up')
        #print(x.shape)
        x = self.dconv_up2(x)
        #print('after upsample layer 2 up')
        #print(x.shape)
        
        # Layer 1 up
        x=self.upconv1(x)
        #print('after upsample layer 1up')
        #print(x.shape)
        x = torch.cat([x, conv1], dim=1)
        #print('after cat layer 1up')
        #print(x.shape)
        x = self.dconv_up1(x)
        #print('after upsample layer 1 up')
        #print(x.shape)
        
        
        # Last layer (output layer)
        out = self.conv_last(x)
        #print('shape of output')
        #print(out.shape)
        
        return out
        
          
                

