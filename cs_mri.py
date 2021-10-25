import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
from tqdm import tqdm
import scipy.io as sio
from datetime import datetime
from skimage.measure import compare_ssim as ssim

from iterAlgs import *
from utils.util import *
from regularizers.robjects import *
from dataFidelities.CSMRIClass_torch import CSMRIClass


with open('./configs/config_mri.json') as File:
    config = json.load(File)

# set the random seed
np.random.seed(128)

# general settings
cs_ratio = config["dataset"]['CS_ratio']
IMG_Patch = config['dataset']['IMG_Patch']

# Read Image path
filepaths_test = get_image_paths(config["dataset"]['valid_datapath'])

# Load CS Sampling Matrix: phi
Phi_data_Name = '%s/mask_%d.mat' % (config["dataset"]['Phi_datapath'], cs_ratio)
Phi_input = sio.loadmat(Phi_data_Name)['mask_matrix']
mask_matrix = torch.from_numpy(Phi_input).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.cuda()

#####################################
#               Valid               # 
#####################################

is_improve = False
ImgNum = len(filepaths_test)

if is_improve:
    index_2run = [2]
else:
    index_2run = range(1, ImgNum+1)

####################################################
####                   PnP/RED                   ###
####################################################
## Regularization parameters
taus_pnp_ar = [0.43850500161952, 0.493710410169959]
taus_pnp_denoising = [1.04482737427516, 1.065167839809401]
taus_red_denoising  = [0.068883707497266, 0.264257381571195]

print('Start Running PnP/RED!')
# TODO: Try Nesterov's acceleration.
for img_no in index_2run:
  
    imgName = filepaths_test[img_no-1]
    print("==============================================================================\n")
    print("img_no: [%d]" % img_no)
    print('imgName: ', imgName)
    print("\n==============================================================================")
    print("")
    Img = cv2.imread(imgName, 0)
    Icol = Img.reshape(1, 1, 256, 256) / 255.0

    Img_output = Icol

    batch_x = torch.from_numpy(Img_output).type(torch.FloatTensor).cuda()
    dObj = CSMRIClass(batch_x, mask)

    ####################################################
    ####                  PnP (AR)                   ###
    ####################################################
    print('Start Running PnP (AR)!')

    tau = taus_pnp_ar[img_no-1]

    alpha = config['pnp(AR)']['alpha']
    cs_ratio_ar = config['pnp(AR)']['cs']
    numIter = config['pnp(AR)']['numIter']
    gamma = config['pnp(AR)']['gamma_inti']

    rObj = DnCNNClass(config, model_path=config['pnp(AR)']['model_path'], cs=cs_ratio_ar)
    save_path = "./results/cs_mri/CS=%d/PnP-AR/img_%d_arcs=%d_sigma_0" %(cs_ratio, img_no, cs_ratio_ar)
    #-- Reconstruction --# 
    recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma, accelerate=False, mode='PROX', useNoise=False, 
                            is_save=True, save_path=save_path, xtrue=[Icol.squeeze(), batch_x], xinit='Atb', save_iter=1000, clip=True, tau=tau, alpha=alpha)
    ####################################################
    ####               PnP （denoising)              ###
    ####################################################
    print('Start Running PnP (denoising)!')
    tau = taus_pnp_denoising[img_no-1]

    sigma = config['pnp(denoising)']['sigma']
    alpha = config['pnp(denoising)']['alpha']
    numIter = config['pnp(denoising)']['numIter']
    gamma = config['pnp(denoising)']['gamma_inti']

    rObj = DnCNNClass(config, model_path=config['pnp(denoising)']['model_path'], sigma=sigma)
    save_path = "./results/cs_mri/CS=%d/PnP-denoising/img_%d_sigma_%d" %(cs_ratio, img_no, sigma)
    #-- Reconstruction --# 
    recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma, accelerate=True, mode='PROX', useNoise=False, 
                            is_save=True, save_path=save_path, xtrue=[Icol.squeeze(), batch_x], xinit='zeros', save_iter=1000, clip=True, tau=tau, alpha=alpha)
    
    ####################################################
    ####               RED （denoising)              ###
    ####################################################
    print('Start Running RED (denoising)!')

    tau = taus_red_denoising[img_no-1]
    
    sigma = config['red(denoising)']['sigma']
    alpha = config['red(denoising)']['alpha']
    numIter = config['red(denoising)']['numIter']
    gamma = config['red(denoising)']['gamma_inti']

    rObj = DnCNNClass(config, model_path=config['red(denoising)']['model_path'], sigma=sigma)
    save_path = "./results/cs_mri/CS=%d/RED-denoising/img_%d_sigma_%d" %(cs_ratio, img_no, sigma)
    #-- Reconstruction --# 
    recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma, accelerate=True, mode='RED', useNoise=False, 
                            is_save=True, save_path=save_path, xtrue=[Icol.squeeze(), batch_x], xinit='zeros', save_iter=1000, clip=True, tau=tau, alpha=alpha)