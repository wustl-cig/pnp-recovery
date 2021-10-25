from __future__ import print_function, division, absolute_import, unicode_literals

import sys
import os
import shutil
import math
import numpy as np
import logging
import torch

from utils.net_mode import *
from utils.util import *
from abc import ABC, abstractmethod
from collections import OrderedDict


############## Basis Class ##############

class RegularizerClass(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prox(self,z,step,pin):
        pass

    @abstractmethod
    def eval(self,z,step,pin):
        pass

    def name(self):
        pass

############## Regularizer Class ##############

class NNClass(RegularizerClass):
    tau = 0
    def __init__(self, sigSize):
        self.sigSize = sigSize

    def init(self):
        p = np.zeros(self.sigSize)
        return p

    def eval(self,x):
        return 0
    
    def prox(self, z, step, pin):
        return np.clip(z,0,np.inf), pin

    def name(self):
        return 'NN'

class ZeroClass(RegularizerClass):
    def __init__(self, sigSize):
        self.sigSize = sigSize

    def init(self):
        p = np.zeros(self.sigSize)
        return p

    def eval(self,x):
        return 0
    
    def prox(self, z, step, pin):
        return 0.0, pin

    def name(self):
        return 'Zero'

class L1Class(RegularizerClass):
    def __init__(self, sigSize, tau):
        self.sigSize = sigSize
        self.tau = tau
    
    def init(self):
        p = np.zeros(self.sigSize)
        return p

    def eval(self,x):
        r = self.tau * np.linalg.norm(x.flatten('F'), 1)
        return r
    
    def prox(self,z,step,pin):
        norm_z = np.absolute(z)
        amp = max(norm_z-step*self.tau, 0)
        norm_z[norm_z <= 0] = 1
        x = np.multiply(np.divide(amp, norm_z), z)
        pout = pin
        return x,pout

    def name(self):
        return 'L1'


class DnCNNClass(RegularizerClass):
    """
    A DnCNN implementation
    """
   
    def __init__(self, config:dict, model_path=None, cs=None, sigma=2, img_channel=1,sp=1):

        self.network = net_model(net_mode=config['cnn_model']['net_mode'], config=config['cnn_model'], sp=sp, num_input_channels=img_channel).cuda()
        if sp==1:
            
            if cs is not None:
                # Load AR
                checkpoint = torch.load(os.path.join(model_path, "cs_%d_sigma_0.pkl"%(cs)))
                self.network.load_state_dict(checkpoint,strict=True)
            else:
                # Load denoiser
                checkpoint = torch.load(os.path.join(model_path, "sigma_%.1f.pth"%(sigma)))
                self.network.load_state_dict(checkpoint['model_state_dict'],strict=True)
        elif sp==2:
            self.network = nn.DataParallel(self.network).cuda()

            checkpoint = torch.load(os.path.join(model_path, "sigma_%.1f.pth"%(sigma)))
            self.network.load_state_dict(checkpoint,strict=True)
        self.lst_vars = self._get_vars()

    def _get_vars(self):
        lst_vars = []
        num_count = 0
        for para in self.network.parameters():
            num_count += 1
            print('Layer %d' % num_count)
            print(para.size())
            lst_vars.append(para)
        return lst_vars    

    
    def init(self):
        p = np.zeros([self.nx, self.ny, self.nz])
        return p

    def red(self, s, pin, useNoise, clip=False, tau=None):

        if clip:
            s[s<=0] = 0
        else:
            pass

        if len(s.shape) == 4:
            # reshape
            self.network.eval()
            with torch.no_grad():
                batch_pre = self.network(s)                                
        else:
            print('Incorrect s.shape')
            exit()

        if useNoise:
            noise = batch_pre
        else:
            noise = s - batch_pre

        return noise, pin

    def prox(self, s, pin, clip=False, tau=1):

        if clip:
            s[s<=0] = 0
        else:
            pass
        
        if len(s.shape) == 4:
            self.network.eval()
            with torch.no_grad():
                batch_pre = 1/tau * self.network(tau*s) 
        else:
            print('Incorrect s.shape')
            exit()
        # batch_pre = s - batch_pre
        return batch_pre, pin

    def denoise(self, s, clip=False, useNoise=False):

        if clip:
            s[s<=0] = 0
        else:
            pass

        if len(s.shape) == 4:
            # reshape
            self.network.eval()
            with torch.no_grad():
                batch_pre = self.network(s)                                
        else:
            print('Incorrect s.shape')
            exit()

        if useNoise:
            noise = batch_pre
        else:
            noise = s - batch_pre

        return noise

    def eval(self, x):
        return 0

    
    def name(self):
        return 'DnCNN'
