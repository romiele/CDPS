# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 11:04:39 2026
    Integrate copyright
@author: Roberto Miele
"""
__author__ = "Roberto Miele"
__copyright__ = ""
__license__ = "Public Domain"

import argparse
import sys
pc = 'rmiele'
def CDPS_pars() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser()
    
    # General
    parser.add_argument("--n_samples",       default = 1, help = 'Number of samples to generate')
    parser.add_argument("--seed",            default = 1,  help = 'Seed for reproducibility') 
    parser.add_argument("--code_dir",        default = f'.../path/to/CDPS/', help='Where the CDPS is stored') 
    parser.add_argument("--work_dir",        default = './', help = 'Working folder path')
    parser.add_argument("--input_dir",       default = './Input', help = 'Working folder path')
    parser.add_argument("--desc_out",        default = 'Out_folder_description', help = 'Description for the save directory')
    #[[3.3114,1.0304],[0.1508, 0.0829]]
    # Predicted model parameters
    parser.add_argument("--shape",           default = [1, 48, 152], help = '[n. of properties, y, x] of generated realizations')
    parser.add_argument("--mean_std",        default = [[3.116826,1.215307856357392]], help = 'Mean and std of the TI, if the network output is standardized else None')
    parser.add_argument("--min_max",         default = None, help = 'Min and Max of data for each modelled property')
    
    # Denoiser model parameters
    parser.add_argument("--net_dir",         default = f'C:/Users/{pc}/OneDrive - Université de Lausanne/Codes/Modeling/EDM_Karras/', help = '(path) Directory where diffusion model is stored') 
    parser.add_argument("--net_snap_f",      default = '/save/Bumberg/network-snapshot-004784.pkl', help = '(Valid for this test code) Snapshot/Checkpoint of the trained network') 
    parser.add_argument("--device",          default = 'cuda', help = 'CPU/GPU Device')
    parser.add_argument("--denoiser_C_f",    default = 'xhat0_covar', help = 'File (to read or write) of the denoiser error (sigmas)')
    parser.add_argument("--dataset_dir",     default = f'C:/Users/{pc}/OneDrive - Université de Lausanne/Codes/TI_Data/Bumberg', help = 'If evaluating the denoising error, training data folder is required')
    parser.add_argument("--n_steps",         default = 32, help = 'Number of denoising steps to generate a realization (minimum required suggested: linear=32; nonlinear=250)')
    parser.add_argument("--sigma_max",       default = 80, help = 'Max noise in noisy vector')
    parser.add_argument("--rho",             default = 7, help = 'Determines the noise schedule, 1 is linear. 7 is the optimal exponential trend from Karras EDM paper')  
    parser.add_argument("--Heun_sampling",   default = False, help = '2nd order Heun correction of the ODE evaluation True/False')  
    parser.add_argument("--subsample_Ct",    default= [0,.25,.4,.6,.8, 1], help = 'List e.g., [0,.125,.3,.5,.65,.8, 1] or None') #
    
    #conditioning data parameters
    parser.add_argument("--RPM",             default = 'det_porores', help = 'None or a rock phyisics model: "det_porores", --- ')
    
        #Hard data (direct observations)
    parser.add_argument("--HD_cond",         default = False, help = 'Hard data conditioning True/False')
    parser.add_argument("--HD_idx",          default = [0], help = 'Index(es) of subsurface property being conditioned on HD; "None" for all')
    parser.add_argument("--obs_dir",         default = '/',  help = 'Directory where conditioning data (dobs) is stored')
    parser.add_argument("--HD_file",         default = None, help = 'None or Observed Hard Data filename')
    parser.add_argument("--HD_error",        default = None, help = 'Sigma for [n properties]')
    parser.add_argument("--HD_ftype",        default = None, help = '"GSLIB" or "Vector": GSLIB file including x, y info or a vector with observations saved as numpy or torch')
    parser.add_argument("--HD_mask",         default = None, help = 'If HD_ftype is not GSLIB, provide a mask that indicates the (sparse) location of the observed data')
    
        #Downsampling of ERT data
    parser.add_argument("--DS_cond",         default= True, help = 'Downsampling of ERT data True/False')
    parser.add_argument("--DS_idx",          default = [0], help = 'Index(es) of subsurface property being conditioned on ERT; "None" for all')
    parser.add_argument("--mest_f",          default= 'log_mest_wa.pt', help = '(filename) Estimated parameters distribution')
    parser.add_argument("--MRM_f",           default= 'MRM_wa.pt', help = '(filename) Resolution matrix')
    parser.add_argument("--MCM_f",           default= 'MCM_wa.pt', help = '(filename) Posterior covariance matrix')
    parser.add_argument("--invCtot_f",       default= 'inverse_Ctot', help = '(filename) Inverse of full covariance matrix')
   
        #Geophyisical data (probabilistic)
    parser.add_argument("--PH_cond",         default = 'ERT', help = 'Geophysical data conditioning: None, ERT, Seismic')
    parser.add_argument("--PH_idx",          default = [0], help = 'Index(es) of subsurface property being conditioned on HD; "None" for all')
    parser.add_argument("--type_fm",         default = 'operator', help = 'Available forward operators: Operator, Fullstack or ERT')

            #Fullstack seismic parameters
    parser.add_argument("--dobs_geophD",     default = None, help = 'None or Observed geophysical data')
    parser.add_argument("--geophD_abs_error",default = .05, help= 'Absolute noise sigma')
    parser.add_argument("--geophD_rel_error",default = 1,   help= 'Relative noise sigma')
    parser.add_argument("--operator_file",   default = None, help = 'None or file containing a linear operator (design matrix)')
    parser.add_argument("--wavelet_file",    default = 'wavelet.asc', help = 'Assumed source wavelet')
    
            #ER data setting
    parser.add_argument("--elec_i",          default = -30, help = 'x location (cells) of the first electrode')
    parser.add_argument("--elec_f",          default = 30, help = 'x location (cells) of the last electrode')
    parser.add_argument("--spacing",         default = 1, help = 'Spacing between electrodes')
    parser.add_argument("--boundary",        default = 20, help = 'HALF of the lateral boundary')
    parser.add_argument("--scheme",          default = 'wa', help = 'ER scheme to use (follows Pygimli syntax)')
    
    args =  parser.parse_args()
    return args

if __name__=='__main__':
    args = CDPS_pars() #take parameters
    
    sys.path.append(args.code_dir)
    from CDPS import CDPS_Inversion

    CDPS = CDPS_Inversion(args) #initialize
    res, wrmse = CDPS()


