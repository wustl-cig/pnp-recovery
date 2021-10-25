'''
Class for quadratic-norm on subsampled 2D random measurements
Based on public avaibale ISTA-Net+ 
'''
import torch
import numpy as np
import scipy.io as sio
import math
import sys
import decimal

from utils.util import *

class CSClass(object):
    def __init__(self, x, batch_x ,batch_y, Phi, Qinit, IMG_patch):

        self.x = x
        self.Phix = batch_y  
        self.batch_x = batch_x
        self.IMG_patch = IMG_patch
        
        self.Phi = Phi
        self.Qinit = Qinit

        self.PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        self.PhiTb = torch.mm(batch_y, Phi)     

        [_, self.row, self.col, self.Ipad, self.row_new, self.col_new] = imread_CS_torch_color(self.x, IMG_patch)

    def size(self):
        C, H, W = self.x.shape
        return C, H, W

    def get_init(self, mod='Qinti'):

        C, H, W = self.size()
        if mod == "Qinti":
            init = torch.mm(self.Phix, torch.transpose(self.Qinit, 0, 1))
            init = col2im_CS_torch_color(init.transpose(1,0), self.row, self.col, self.row_new, self.col_new)
        else:
            init = torch.zeros(self.x.size(), dtype=torch.float32)
        
        
        init = init.view(-1, C, H, W)

        return init
    
    def grad(self, x, step=None):
        C, H, W = self.size()
        [_, _, _, Ipad, _, _] = imread_CS_torch_color(x.squeeze(), self.IMG_patch)
        x = img2col_torch_color(Ipad, self.IMG_patch).transpose(1,0)

        v = torch.mm(x, self.PhiTPhi) - self.PhiTb

        v = col2im_CS_torch_color(v.transpose(1,0), self.row, self.col, self.row_new, self.col_new)
        v = v.view(-1, C, H, W)
        return v

    def data_eval(self, x):
        C, H, W = self.size()
        [_, _, _, Ipad, _, _] = imread_CS_torch_color(x.squeeze(), self.IMG_patch)
        x = img2col_torch_color(Ipad, self.IMG_patch).transpose(1,0)
        data_dist = torch.norm(torch.mm(x, torch.transpose(self.Phi, 0, 1)).flatten() - self.Phix.flatten())**2
        return data_dist
    
    def draw(self,x):
        pass
