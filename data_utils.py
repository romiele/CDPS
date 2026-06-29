# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 11:28:29 2026
    Include:
    - Data loaders
    - Normalization
    - Rescaling 
    - Standardization
    - Flipping

@author: rmiele
"""
import numpy as np
import torch

class Compose_datat:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

def crop_padded_area(array,paddedshp):
    """removes the additional 0-padded area that is included just for the Karras Unet"""
    h_pad, w_pad = paddedshp
    return array[..., :(-h_pad or None), :(-w_pad or None)]

def norm(array):
    """ rescales network output [-1,1] support to [0,1]"""
    return (array+1)*.5


# def rescale_minmax(array, min_max):
#     """Rescales network output from [-1,1] to target range"""
#     array = (array + 1) * 0.5

#     minv = min_max[:, 0]
#     maxv = min_max[:, 1]

#     return array * (maxv - minv)[None, :, None, None] + minv[None, :, None, None]

def rescale_minmax(array, min_max):
    """ rescales network output from [-1,1] to m_est support"""
    array = (array+1)*.5
    for i in range(array.shape[1]):
        array[:,i] = array[:,i]*(min_max[i][1]-min_max[i][0]) + min_max[i][0]
    return array


# def rescale_meanstd(array, meanstd):
#     mean = meanstd[:, 0]
#     std = meanstd[:, 1]
#     return array * (2 * std)[None, :, None, None] + mean[None, :, None, None]

def rescale_meanstd(array, meanstd):
    for i in range(array.shape[1]):
        array[:,i] = (array[:,i]*(meanstd[i][1])+meanstd[i][0])

    return array


def flip(array):
    return torch.flip(array, dims=[-2])


def interp_ctot(CDPS, i, invcov, cache={}):
    """ 
    Linearly interpolates the "undersampled" Inverse Full Covariance matrix
    for the sampling step where this is not sampled
    """
    # import matplotlib.pyplot as plt 

    n = CDPS.t_steps[i]
    
    if n in CDPS.t_Csteps:
        idx = torch.where(CDPS.t_Csteps == n)[0][0].item()
        
        # plt.figure()
        # plt.title(f'index {idx} - that is sigma={CDPS.t_steps[i]:.3f}')
        # plt.imshow(np.diag(invcov[idx][0]).reshape(32,32))
        # plt.colorbar()
        # plt.show()
        return invcov[idx]

    idx_min = torch.searchsorted(-CDPS.t_Csteps, -n) - 1

    # cache key
    key0, key1 = idx_min.item(), idx_min.item() + 1

    if key0 not in cache:
        cache[key0] = invcov[key0]
    if key1 not in cache:
        cache[key1] = invcov[key1]

    v0 = cache[key0]
    v1 = cache[key1]

    n0 = CDPS.t_Csteps[idx_min]
    n1 = CDPS.t_Csteps[idx_min+1]
    r = ((n - n0) / (n1 - n0)).item()
    
    invmat = v0 * (1 - r) + v1 * r
    
    # plt.figure()
    # plt.title(f'That is the approx at sigma={CDPS.t_steps[i]:.3f}\nInterp between {CDPS.t_Csteps[idx_min]:.3f} and {CDPS.t_Csteps[idx_min+1]:.3f} ')
    # plt.imshow(np.diag(invmat[0]).reshape(32,32))
    # plt.colorbar()    
    # plt.show()

    return invmat


def prec_inverse(CDPS, args):
    import os
    from tqdm import tqdm
    # import matplotlib.pyplot as plt 

    """
    Pre-computes the inverse of the full covariance matrix 
    for each sampling step or a subset (if subsample_Ctot==True)
    Saves a numpy memmap to save memory
    """
    if CDPS.condtype[-1]=='DS_cond': 
        shape = (len(CDPS.t_Csteps),1,
                 CDPS.shp[-1]*CDPS.shp[-2],
                 CDPS.shp[-1]*CDPS.shp[-2]) 
        
    else:
        shape = (len(CDPS.t_Csteps),CDPS.shp[1],
                 CDPS.shp[-1]*CDPS.shp[-2],
                 CDPS.shp[-1]*CDPS.shp[-2]) 

    n_split=8
    big_inverse_cov = np.save(CDPS.save_dir+'/'+CDPS.invCtot_f, 
                          np.zeros(shape).astype(np.float16))
    
    del big_inverse_cov
    big_inverse_cov = np.load(CDPS.save_dir+'/'+CDPS.invCtot_f, 
                          mmap_mode = 'r+')
    
    pbar = tqdm(range(len(CDPS.t_Csteps)), desc= 'Computing inverse cov.')
    for p in range(shape[1]):
        idx_p = CDPS.idx_p[p]
        for i in pbar:
            #propagate the denoiser error
            temp = CDPS.full_covariance(torch.from_numpy(CDPS.cov_xhat0[i,idx_p]).to(CDPS.device).double(), p=p)
            temp = torch.inverse(temp)
            big_inverse_cov[i] = temp.cpu().numpy()
            del temp

    return big_inverse_cov