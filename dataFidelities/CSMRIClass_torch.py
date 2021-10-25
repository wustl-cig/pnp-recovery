'''
Class for quadratic-norm on subsampled 2D Fourier measurements
Based on public avaibale ISTA-Net+ 
'''
import torch
import numpy as np
import scipy.io as sio
import math
import sys
import decimal

from utils.util import *

class CSMRIClass(object):
    def __init__(self, x, mask):

        self.x = x
        self.mask = mask
        self.PhiTb = self.FFT_Mask_ForBack(x, self.mask)

    def size(self):
        B,C,H, W = self.x.shape
        return B, C, H, W

    def get_init(self, mod='Qinti'):

        B, C, H, W = self.size()
        if mod == "Qinti":
            init = self.PhiTb
        else:
            init = torch.zeros(self.x.size(), dtype=torch.float32)
        
        init = init.view(-1, 1, H, W)

        return init
    
    def grad(self, x, step=None):
        B,C,H, W = self.size()
        v = self.FFT_Mask_ForBack(x, self.mask) - self.PhiTb
        v = v.view(-1, 1, H, W)
        return v
    
    def draw(self,x):
        pass
    
    @staticmethod
    def FFT_Mask_ForBack(x, mask):
        x_dim_0 = x.shape[0]
        x_dim_1 = x.shape[1]
        x_dim_2 = x.shape[2]
        x_dim_3 = x.shape[3]
        x = x.view(-1, x_dim_2, x_dim_3, 1)
        y = torch.zeros_like(x)
        z = torch.cat([x, y], 3)
        fftz = torch.fft(z, 2)
        z_hat = torch.ifft(fftz * mask, 2)
        x = z_hat[:, :, :, 0:1]
        x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
        return x
     