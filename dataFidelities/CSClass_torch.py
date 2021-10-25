'''
Class for quadratic-norm on subsampled 2D CS measurements
Based on ISTA-Net+ Pytorch code.
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

        [_, self.row, self.col, self.Ipad, self.row_new, self.col_new] = imread_CS_torch(self.x, IMG_patch)

    def size(self):
        H, W = self.x.shape
        return H, W

    def get_init(self, mod='Qinti'):

        H, W = self.size()
        if mod == "Qinti":
            init = torch.mm(self.Phix, torch.transpose(self.Qinit, 0, 1))
        else:
            init = torch.zeros(self.x.size(), dtype=torch.float32)
        
        init = col2im_CS_torch(init.transpose(1,0), self.row, self.col, self.row_new, self.col_new)
        init = init.view(-1, 1, H, W)

        return init
    
    def grad(self, x, step=None):
        H, W = self.size()
        [_, _, _, Ipad, _, _] = imread_CS_torch(x.squeeze(), self.IMG_patch)
        x = img2col_torch(Ipad, self.IMG_patch).transpose(1,0)

        v = torch.mm(x, self.PhiTPhi) - self.PhiTb

        v = col2im_CS_torch(v.transpose(1,0), self.row, self.col, self.row_new, self.col_new)
        v = v.view(-1, 1, H, W)
        return v


    def data_eval(self, x):
        H, W = self.size()
        [_, _, _, Ipad, _, _] = imread_CS_torch(x.squeeze(), self.IMG_patch)
        x = img2col_torch(Ipad, self.IMG_patch).transpose(1,0)
        data_dist = torch.norm(torch.mm(x, torch.transpose(self.Phi, 0, 1)).flatten() - self.Phix.flatten())**2
        return data_dist
    
    def draw(self,x):
        pass
    
    @staticmethod
    def Qinit(Phi_input, cs_ratio):
        # Generate Qinit, brow from ISTA-Net+
        Training_data = sio.loadmat("./.") # replace with your path
        Training_labels = Training_data["./."] # replace with your name

        X_data = Training_labels.transpose()
        Y_data = np.dot(Phi_input, X_data)
        Y_YT = np.dot(Y_data, Y_data.transpose())
        X_YT = np.dot(X_data, Y_data.transpose())
        Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
        del X_data, Y_data, X_YT, Y_YT
        # sio.savemat("Initialization_Matrix_%d.mat"%(cs_ratio), {'Qinit': Qinit})    
        return Qinit
     