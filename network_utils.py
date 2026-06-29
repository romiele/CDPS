# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:55:22 2026
    It contains all the functions related to the network, including
    - Loading the EDM from karras (check the way it pickles and their torch_utils)
    - Evaluate the network error
    - ?
    
@author: rmiele
"""
import pickle
import sys
import torch
import os
import numpy as np

def import_network(args):
    sys.path.append(args.net_dir) #For EDM, it is necessary to use dnnlib folder from the original code
    try:
        #load network
        with open(args.net_dir+args.net_snap_f, 'rb') as f:
            net = pickle.load(f)['ema'].to(args.device)
    except ModuleNotFoundError as e: 
        print(str(e) + ' in network dir [net_dir].')
        print('torch_utils folder from Karras et al.(2022): https://github.com/NVlabs/edm/tree/main/')
        
    return net



def get_dataloader(args):
    sys.path.append(args.net_dir) #For EDM, it is necessary to use dnnlib folder from the original code
    datafolder = args.dataset_dir.split('/')[-1]
    
    if 'Geostatistics' in datafolder :
        from training.dataset import Geost_dataset
        dataset = Geost_dataset(args.dataset_dir, [args.shape[-2],args.shape[-1]], args.shape[1])
    
    elif datafolder == 'Bumberg' :
        from training.dataset import FaciesSet_parse3D
        dataset = FaciesSet_parse3D(args.dataset_dir, 
                                    [args.shape[-2],args.shape[-1]],
                                    args.shape[1])
    
    elif datafolder == 'Synthetic_channels' :
        from training.dataset import FaciesSet
        dataset = FaciesSet(args.dataset_dir, [args.shape[-2],args.shape[-1]], args.shape[1])

    return torch.utils.data.DataLoader(dataset, batch_size=300, shuffle=True)



def compute_Cxhat0(args, CDPS):
    # import matplotlib.pyplot as plt 

    from tqdm import tqdm
    loader = get_dataloader(args)
    for i, test_data in enumerate(loader):
        test_data = test_data[0].to(args.device)
        break
    
    transformations = CDPS.HD_transf if CDPS.condtype[-1] == 'HD_cond' else (CDPS.FW_transf if CDPS.condtype[-1]=='DS_cond' else None)
    
    if CDPS.condtype[-1]=='DS_cond': 
        shape = (len(CDPS.t_Csteps),1,
                 CDPS.shp[-1]*CDPS.shp[-2],
                 CDPS.shp[-1]*CDPS.shp[-2]) 
        
    else:
        shape = (len(CDPS.t_Csteps),CDPS.shp[1],
                 CDPS.shp[-1]*CDPS.shp[-2],
                 CDPS.shp[-1]*CDPS.shp[-2]) 

    n_split=8
    
    covar_xhat0 = np.save(CDPS.input_dir+'/'+CDPS.denoiser_C_fn, 
                          np.zeros(shape).astype(np.float16))
    
    del covar_xhat0
    covar_xhat0 = np.load(CDPS.input_dir+'/'+CDPS.denoiser_C_fn, 
                          mmap_mode = 'r+')
    
    sigma_list = np.zeros((len(CDPS.t_Csteps), 2))
    
    split = test_data.shape[0]//n_split
    
    D = torch.zeros(test_data.shape[0], shape[1], CDPS.shp[-1]*CDPS.shp[-2])
    for i in tqdm(range(len(CDPS.t_Csteps[:-1])), desc= 'Error at time-step'):
        for j in range(split): #split for memory limits
            idx_i = j*n_split; idx_f = j*n_split+n_split
            
            target = test_data[idx_i:idx_f].clone()
            target += torch.randn_like(target).to(args.device)*CDPS.t_Csteps[i] #Noisy input to the network
            
            pred = CDPS.net(target, CDPS.t_Csteps[i], None) #estimate denoised img
            if CDPS.has_lbl: 
                label = pred[1]
                pred = pred[0]

            target = transformations(test_data[idx_i:idx_f].clone())
            if args.RPM is not None: target = CDPS.rpm_solver(target) #error estimated AFTER RPM 

            pred = transformations(pred)
            if args.RPM is not None: pred = CDPS.rpm_solver(pred) #error estimated AFTER RPM 
            
            if target.ndim<4:
                pred = pred[:,None,:]
                target = target[:,None,:]
                
            for p in range(shape[1]): #accounts for multiple properties (huge load on hard drives)
                D[idx_i:idx_f,p] = (target[:,p] - pred[:,p]).flatten(1) #store the errors for the "minibatch"
                
        D = D - D.mean(0)[None,:] #remove the bias
        for p in range(shape[1]):
            covar_xhat0[i,p] = ((D[:,p].T @ D[:,p])/D.shape[0]).detach().cpu()
            sigma_list[i][p] = np.mean(np.sqrt(np.diag(covar_xhat0[i,p])))
        
        # plt.figure()
        # plt.title(f'C_den bias {i}')
        # plt.imshow(D.mean(0).reshape(48,152))
        # plt.colorbar()
        # plt.figure()
        # plt.title(f'C_den diag {i}')
        # plt.imshow(np.diag(covar_xhat0[i,0]).reshape(48,152))
        # plt.colorbar()
        # plt.figure()
        # plt.title(f'C_den 100 {i}')
        # plt.imshow(covar_xhat0[i,0,:,100].reshape(48,152))
        # plt.colorbar()
        # plt.show()
    return covar_xhat0




