# library
import os
import shutil
import scipy.io as sio
import numpy as np
import warnings
import time
# scripts
from tqdm import tqdm
from utils.util import *

######## Iterative Methods #######

def apgmEst(dObj, rObj, 
            numIter=100, step=100, accelerate=False, mode='RED', useNoise=True, 
            is_save=True, save_path='result', xtrue=None, xinit=None, save_iter=2000, clip=False, tau=1, alpha=1):
    """
    Plug-and-Play Accelerated Proximal Gradient Method with switch for PGM and SPGM
    
    ### INPUT:
    dObj       ~ data fidelity term, measurement/forward model
    rObj       ~ regularizer term
    numIter    ~ total number of iterations
    accelerate ~ use APGM or PGM
    mode       ~ RED update or PROX update
    useNoise   ~ CNN predict noise or image
    step       ~ step-size
    is_save    ~ if true save the reconstruction of each iteration
    save_path  ~ the save path for is_save
    xtrue      ~ the ground truth of the image, for tracking purpose

    ### OUTPUT:
    x     ~ reconstruction of the algorithm
    outs  ~ detailed information including cost, snr, step-size and time of each iteration

    """
    
    ##### HELPER FUNCTION #####
    evaluateSnr = lambda xtrue, x: 20*np.log10(np.linalg.norm(xtrue.flatten('F'))/np.linalg.norm(xtrue.flatten('F')-x.flatten('F')))
    ##### INITIALIZATION #####
    
    # initialize save foler
    if is_save:
        abs_save_path = os.path.abspath(save_path)
        if os.path.exists(save_path):
            print("Removing '{:}'".format(abs_save_path))
            shutil.rmtree(abs_save_path, ignore_errors=True)
        # make new path
        print("Allocating '{:}'".format(abs_save_path))
        os.makedirs(abs_save_path)
    
    #initialize info data
    if xtrue is not None:
        xtrueSet = True
        snr_iters = []
        psnr_iters = []
    else:
        xtrueSet = False

    timer = []

    # initialize variables
    if xinit == 'zeros':
        xinit = dObj.get_init()
        xinit = torch.zeros(xinit.shape, dtype=xinit.dtype).cuda()
    else:    
        xinit = dObj.get_init()

    x = xinit
    s = x  # gradient update
    step = torch.tensor(step, dtype=torch.float32)
    t = torch.tensor(1., dtype=torch.float32)  # controls acceleration
    p = torch.zeros(xinit.shape, dtype=torch.float32)# dual variable for TV
    pfull = torch.zeros(xinit.shape, dtype=torch.float32)# dual variable for TV

    ##### MAIN LOOP #####
    run_t = time.time()
    for indIter in tqdm(range(numIter)):
        # get gradient
        g_dobj = dObj.grad(s)
        if mode == 'RED':
            g_robj, _ = rObj.red(s, p, useNoise, clip=False)
            xnext = s - step*(g_dobj + tau*g_robj)
            if clip:
                xnext[xnext<=0] = 0
        elif mode == 'PROX':
            vnext = s - step*g_dobj
            Px, p = rObj.prox(vnext, p, clip=clip, tau=tau)   # clip to [0, inf], alpha averaged operator
            xnext = (1 - alpha) * vnext + alpha * Px
        elif mode == 'GRAD':
            xnext = vnext
            xnext[xnext<=0] =0
        else:
            print("No such mode option")
            exit()

        if xtrueSet:
            psnr_iters.append(compare_psnr(x.squeeze(), xtrue[1]).item())

        # acceleration
        if accelerate:
            tnext = 0.5*(1+torch.sqrt(1+4*t*t))
        else:
            tnext = 1
        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # update
        t = tnext
        x = xnext

        # save & print
        if is_save and (indIter+1) % (save_iter/2) == 0:
            img_save = np.clip(x.squeeze().cpu().data.numpy(), 0, 1).astype(np.float64)
            if len(img_save.shape)==3 and img_save.shape[0]==3:
                img_save = img_save.transpose([1,2,0])
            save_img(img_save, abs_save_path+'/iter_%d_%.2fdB.tif'%(indIter+1, psnr_iters[indIter]))

    # summarize outs
    x_rec = np.clip(x.squeeze().cpu().data.numpy(), 0, 1).astype(np.float64)

    outs = {
        "psnr": np.array(psnr_iters),
        "recon": x_rec,
        "run_time" : time.time() - run_t
    }

    if is_save:
        save_mat(outs, abs_save_path+'/out.mat'.format(indIter+1))

    return x_rec, outs, np.array(psnr_iters)

