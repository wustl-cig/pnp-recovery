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
from dataFidelities.CSClass_torch import CSClass


with open('./configs/config_nature_gray.json') as File:
    config = json.load(File)

# set the random seed
np.random.seed(128)

# general settings
cs_ratio = config["dataset"]['CS_ratio']
IMG_Patch = config['dataset']['IMG_Patch']

# Read Image path
filepaths_test = get_image_paths(config["dataset"]['valid_datapath'])

# Load CS Sampling Matrix: phi
Phi_data_Name = '%s/phi_0_%d_1089.mat' % (config["dataset"]['Phi_datapath'], cs_ratio)
Phi_input = sio.loadmat(Phi_data_Name)['phi']

# Load Initialization Matrix: Qinit
Qinit_data_Name = '%s/Initialization_Matrix_%d.mat' % (config["dataset"]['Qinit_datapath'], cs_ratio)
Qinit = sio.loadmat(Qinit_data_Name)['Qinit']

# Convert from numpy to torch:
Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Phi = Phi.cuda()
Qinit = Qinit.cuda()

# Load Regularization parameters

parameters = sio.loadmat('./data/nature_gray/parameters.mat', squeeze_me=True)

pnp_ar = parameters['pnp_ar']
pnp_denoising = parameters['pnp_denoising']
red_denoising = parameters['red_denoising']
#####################################
#               Valid               # 
#####################################

is_improve = False
ImgNum = len(filepaths_test)

if is_improve:
    index_2run = []
else:
    index_2run = range(1, ImgNum+1)

####################################################
####                   PnP/RED                   ###
####################################################
print('Start Running PnP/RED!')
# TODO: Try Nesterov's acceleration.
 
for img_no in index_2run:
  
    imgName = filepaths_test[img_no-1]
    print("==============================================================================\n")
    print("img_no: [%d]" % img_no)
    print('imgName: ', imgName)
    print("\n==============================================================================")
    print("")
    Img = cv2.imread(imgName, 1)
    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Img_rec_yuv = Img_yuv.copy()
    Iorg_y = Img_yuv[:,:,0].astype(np.float64)/255.0
    Iorg_y_torch = torch.from_numpy(Iorg_y).type(torch.FloatTensor).cuda()
    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_torch(Iorg_y_torch, IMG_Patch)
    Icol = img2col_torch(Ipad, IMG_Patch).transpose(1,0)
    Img_output = Icol

    batch_x = Img_output
    Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1)).cuda()
    dObj = CSClass(Iorg_y_torch, batch_x, Phix, Phi, Qinit, IMG_Patch)

    # ####################################################
    # ####                  PnP (AR)                   ###
    ####################################################
    print('Start Running PnP (AR)!')

    cs_ratio_ar, tau = pnp_ar[img_no-1]
    
    alpha = config['pnp(AR)']['alpha']
    cs_ratio_ar = config['pnp(AR)']['cs']
    numIter = config['pnp(AR)']['numIter']
    gamma = config['pnp(AR)']['gamma_inti']

    rObj = DnCNNClass(config, model_path=config['pnp(AR)']['model_path'], cs=cs_ratio_ar)
    save_path = "./results/cs_nature_gray/CS=%d/PnP-AR/img_%d_arcs=%d_sigma_0" %(cs_ratio, img_no, cs_ratio_ar)
    #-- Reconstruction --# 
    recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma, accelerate=False, mode='PROX', useNoise=False, 
                            is_save=True, save_path=save_path, xtrue=[Iorg_y, Iorg_y_torch], xinit='Qinti', save_iter=6000, clip=True, tau=tau, alpha=alpha)
    ####################################################
    ####               PnP （denoising)              ###
    ####################################################
    print('Start Running PnP (denoising)!')

    sigma, tau = pnp_denoising[img_no-1]
    
    alpha = config['pnp(denoising)']['alpha']
    numIter = config['pnp(denoising)']['numIter']
    gamma = config['pnp(denoising)']['gamma_inti']

    rObj = DnCNNClass(config, model_path=config['pnp(denoising)']['model_path'], sigma=sigma)
    save_path = "./results/cs_nature_gray/CS=%d/PnP-denoising/img_%d_sigma_%d" %(cs_ratio, img_no, sigma)
    #-- Reconstruction --# 
    recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma, accelerate=False, mode='PROX', useNoise=False, 
                            is_save=True, save_path=save_path, xtrue=[Iorg_y, Iorg_y_torch], xinit='Qinti', save_iter=6000, clip=True, tau=tau, alpha=alpha)
    # ###############################################################################
    # TODO: Using spectral normalization in [23] on DnCNN denoisers -- uncomment the following
    # ###############################################################################
    # rObj = DnCNNClass(config, model_path='./model_zoo/gray_denoisers_L12_sp2', sigma=5, sp=2)
    # save_path = "./results/cs_nature_gray/CS=%d/PnP-denoising_sp2/img_%d_sigma_%d" %(cs_ratio, img_no, 5)
    # #-- Reconstruction --# 
    # recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=3000, step=1.5, accelerate=False, mode='PROX', useNoise=False, 
    #                         is_save=True, save_path=save_path, xtrue=[Iorg_y, Iorg_y_torch], xinit='Qinti', save_iter=6000, clip=True, tau=1.125, alpha=0.2)
    ####################################################
    ####               RED （denoising)              ###
    ####################################################
    print('Start Running RED (denoising)!')

    sigma, tau = red_denoising[img_no-1]
    
    alpha = config['red(denoising)']['alpha']
    numIter = config['red(denoising)']['numIter']
    gamma = config['red(denoising)']['gamma_inti']

    rObj = DnCNNClass(config, model_path=config['red(denoising)']['model_path'], sigma=sigma)
    save_path = "./results/cs_nature_gray/CS=%d/RED-denoising/img_%d_sigma_%d" %(cs_ratio, img_no, sigma)
    #-- Reconstruction --# 
    recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma/(1 + 2*tau), accelerate=False, mode='RED', useNoise=False, 
                            is_save=True, save_path=save_path, xtrue=[Iorg_y, Iorg_y_torch], xinit='Qinti', save_iter=6000, clip=False, tau=tau, alpha=alpha)