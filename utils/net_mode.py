from __future__ import print_function

from models.unet import UNet

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

def net_model(config: dict, sp = 1, num_input_channels = 1, net_mode=3, input_depth=1, pad = 'zero', NET_TYPE = 'skip_depth4'):

    if net_mode == 1:
        net = UNet(in_channels=num_input_channels, out_channels=num_input_channels, 
                   init_features=64)
    elif net_mode == 2:
        if sp==1:
            from models.dncnn_sp1 import DnCNN
            net = DnCNN(image_channels=num_input_channels) 
        elif sp==2:
            from models.dncnn_sp2 import DnCNN
            net = DnCNN(image_channels=num_input_channels)              
    else:
        assert False
    return net