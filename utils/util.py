"""
Parts of the codes borrowed and modified from public available source:
1, https://github.com/cszn/KAIR
2, https://github.com/jianzhangcs/ISTA-Net-PyTorch
"""

import torch
import torch.nn as nn

import os
import cv2
import h5py
import math
import shutil
import imageio
import numpy as np
import scipy.io as sio
import scipy.misc as smisc
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from skimage.measure import compare_ssim as ssim

evaluateSnr = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))
# Image file extensions
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def surf(Z, cmap='rainbow', figsize=None):
    plt.figure(figsize=figsize)
    ax3 = plt.axes(projection='3d')

    w, h = Z.shape[:2]
    xx = np.arange(0,w,1)
    yy = np.arange(0,h,1)
    X, Y = np.meshgrid(xx, yy)
    ax3.plot_surface(X,Y,Z,cmap=cmap)
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap=cmap)
    plt.show()


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

'''
# --------------------------------------------
# split large images into small images 
# --------------------------------------------
'''

def patches_from_image(img, p_size=512, p_overlap=64, p_max=800):
    w, h = img.shape[:2]
    patches = []
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=np.int))
        w1.append(w-p_size)
        h1.append(h-p_size)
        for i in w1:
            for j in h1:
                patches.append(img[i:i+p_size, j:j+p_size,:])
    else:
        patches.append(img)

    return np.array(patches, dtype=np.float32)


def imssave(imgs, img_path):
    """ 
    imgs: list, N images of size WxHxC
    """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    for i, img in enumerate(imgs):
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]
        new_path = os.path.join(os.path.dirname(img_path), img_name+str('_{:04d}'.format(i))+'.png')
        cv2.imwrite(new_path, img)


def split_imageset(original_dataroot, taget_dataroot, n_channels=3, p_size=512, p_overlap=96, p_max=800):
    """
    split the large images from original_dataroot into small overlapped images with size (p_size)x(p_size), 
    and save them into taget_dataroot; only the images with larger size than (p_max)x(p_max)
    will be splitted.

    Args:
        original_dataroot:
        taget_dataroot:
        p_size: size of small images
        p_overlap: patch size in training is a good choice
        p_max: images with smaller size than (p_max)x(p_max) keep unchanged.
    """
    paths = get_image_paths(original_dataroot)
    for img_path in paths:
        # img_name, ext = os.path.splitext(os.path.basename(img_path))
        img = imread_uint(img_path, n_channels=n_channels)
        patches = patches_from_image(img, p_size, p_overlap, p_max)
        imssave(patches, os.path.join(taget_dataroot, os.path.basename(img_path)))
        #if original_dataroot == taget_dataroot:
        #del img_path

'''
# --------------------------------------------
# read image from path
# opencv is fast, but read BGR numpy image
# --------------------------------------------
'''
# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE # HxW
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

# --------------------------------------------
# matlab's imwrite
# --------------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

# --------------------------------------------
# get single image of size HxWxn_channles (BGR)
# --------------------------------------------
def read_img(path):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img



'''
# --------------------------------------------
# image format conversion
# --------------------------------------------
# numpy(single) <--->  numpy(uint)
# numpy(single) <--->  tensor
# numpy(uint)   <--->  tensor
# --------------------------------------------
'''


# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(uint)
# --------------------------------------------


def uint2single(img):

    return np.float32(img/255.)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())


def uint162single(img):

    return np.float32(img/65535.)


def single2uint16(img):

    return np.uint16((img.clip(0, 1)*65535.).round())


def optimizeTau(x, algoHandle, taurange, maxfun=20):
    # maxfun ~ number of iterations for optimization
    # fun = lambda tau: -psnr(algoHandle(tau), x)
    fun = lambda tau: -algoHandle(tau)[2]
    tau, fx_all, tall_all = fminbound(fun, taurange[0],taurange[1], xtol = 2e-3, maxfun = maxfun, disp = 3)
    return tau, np.abs(np.array(fx_all)), np.abs(np.array(tall_all))

def save_img(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    img = to_rgb(img)
    imageio.imwrite(path, img.round().astype(np.uint8))


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def to_double(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    return img

def save_mat(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    
    sio.savemat(path, {'img':img})

def h5py2mat(data):
    result = np.array(data)
    print(result.shape)

    if len(result.shape) == 3 and result.shape[0] > result.shape[1]:
        result = result.transpose([1,0,2])
    elif len(result.shape) == 3 and result.shape[1] < result.shape[2]:
        result = result.transpose([2,1,0])
    elif len(result.shape) == 3 and result.shape[1] > result.shape[2]:
        result = result.transpose([2,1,0])        
    print(result.shape)    
    return result

def complex_multiple_torch(x: torch.Tensor, y: torch.Tensor):
    x_real, x_imag = torch.unbind(x, -1)
    y_real, y_imag = torch.unbind(y, -1)

    res_real = torch.mul(x_real, y_real) - torch.mul(x_imag, y_imag)
    res_imag = torch.mul(x_real, y_imag) + torch.mul(x_imag, y_real)

    return torch.stack([res_real, res_imag], -1)

def np2torch_complex(array: np.ndarray):
    return torch.stack([torch.from_numpy(array.real), torch.from_numpy(array.imag)], -1)

def addwgn_torch(x: torch.Tensor, inputSnr):
    noiseNorm = torch.norm(x.flatten() * 10 ** (-inputSnr / 20))

    noise = torch.randn(x.shape[-2], x.shape[-1])
    noise = noise / torch.norm(noise.flatten()) * noiseNorm
    
    rec_y = x + noise

    return rec_y

def compare_snr(img_test, img_true):
    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))

def rsnr_cal(rec,oracle):
    "regressed SNR"
    sumP    =        sum(oracle.reshape(-1))
    sumI    =        sum(rec.reshape(-1))
    sumIP   =        sum( oracle.reshape(-1) * rec.reshape(-1) )
    sumI2   =        sum(rec.reshape(-1)**2)
    A       =        np.matrix([[sumI2, sumI],[sumI, oracle.size]])
    b       =        np.matrix([[sumIP],[sumP]])
    c       =        np.linalg.inv(A)*b #(A)\b
    rec     =        c[0,0]*rec+c[1,0]
    err     =        sum((oracle.reshape(-1)-rec.reshape(-1))**2)
    SNR     =        10.0*np.log10(sum(oracle.reshape(-1)**2)/err)

    if np.isnan(SNR):
        SNR=0.0
    return SNR

def compute_rsnr(x, xhat):
    if len(x.shape) == 2:
        avg_rsnr = rsnr_cal(xhat, x)
    elif len(x.shape) == 3 and x.shape[0] < x.shape[1]:   
        rsnr = np.zeros([1,x.shape[0]])
        for num_imgs in range(0,x.shape[0]):
            rsnr[:,num_imgs] = rsnr_cal(xhat, x)
        avg_rsnr = np.mean(rsnr)
    return avg_rsnr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def copytree(src=None, dst=None, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
                
def data_augmentation(image, mode):
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return out                

def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

def imread_CS_torch(Iorg, block_size):

    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = torch.cat((Iorg, torch.zeros([row, col_pad]).cuda()), dim=1)
    Ipad = torch.cat((Ipad, torch.zeros([row_pad, col+col_pad]).cuda()), dim=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]

def img2col_torch(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = torch.zeros([block_size**2, block_num]).cuda()
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col

def col2im_CS_torch(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = torch.zeros([row_new, col_new]).cuda()
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec

def imread_CS_torch_color(Iorg, block_size):

    [chl, row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = torch.cat((Iorg, torch.zeros([chl, row, col_pad]).cuda()), dim=2)
    Ipad = torch.cat((Ipad, torch.zeros([chl, row_pad, col+col_pad]).cuda()), dim=1)
    [chl, row_new, col_new] = Ipad.shape
    return [Iorg, row, col, Ipad, row_new, col_new]

def img2col_torch_color(Ipad, block_size):
    [chl, row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = torch.zeros([3*block_size**2, block_num]).cuda()
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[:, x:x+block_size, y:y+block_size].reshape([-1])
            count = count + 1
       
    return img_col

def col2im_CS_torch_color(X_col, row, col, row_new, col_new,chl=3):
    block_size = 33
    X0_rec = torch.zeros([chl, row_new, col_new]).cuda()
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[:,x:x+block_size, y:y+block_size] = X_col[:, count].reshape([-1, block_size, block_size])
            count = count + 1
    X_rec = X0_rec[:, :row, :col]
    return X_rec

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compare_mse(img_test, img_true, size_average=True):
    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        img_diff = img_diff.mean()
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)

    return img_diff

def compare_psnr(img_test, img_true, size_average=True, max_value=1):
    return 10 * torch.log10((max_value ** 2) / compare_mse(img_test, img_true, size_average))