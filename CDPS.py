# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 16:55:54 2026
    CDPS
@author: romie
"""
import network_utils as nu
import torch
import numpy as np
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
from network_utils import compute_Cxhat0 
from data_utils import rescale_meanstd, rescale_minmax, \
                       flip, prec_inverse, interp_ctot, \
                       Compose_datat, crop_padded_area
import torch.nn.functional as F

class CDPS_Inversion():
    def __init__(self, args):

        torch.manual_seed(args.seed)
        
        #__________General parameters__________________________________________________
        self.work_dir = args.work_dir
        self.input_dir = args.input_dir
        self.save_dir = self.work_dir+args.desc_out #build meaningful save directories
        os.makedirs(self.save_dir, exist_ok=True)
        self.condtype = []
        self.n_samples = args.n_samples
        self.shp = args.shape; self.shp.insert(0,args.n_samples)
            #adjust the shape of the generated image if it is not divisible by 8 by adding a 0-valued area (then cropped after passing through the UNet)
        self.padded_shp = ((8 - (np.array(self.shp) % 8)) % 8)[2:] #8 is due to the network used by Karras et al., 2022: donwsampling and upsampling are done by a factor of 2**3
        self.minibatch = 1; self.minshp = self.shp.copy(); 
        self.minshp[0]=1; self.minshp[2:]+=self.padded_shp
        self.sec_ord = args.Heun_sampling

        #__________Network parameters__________________________________________________
        self.device = args.device
        self.net = nu.import_network(args) #load network with pickle
        self.shp = args.shape
        self.has_lbl = args.has_lbl
        
               #Diffusion sampling
        self.sigma_min = max(0.002, self.net.sigma_min)
        self.sigma_max = min(args.sigma_max, self.net.sigma_max)
        self.rho = args.rho
        self.n_steps = args.n_steps
        
               #Time step discretization
        step_indices = torch.arange(args.n_steps, dtype=torch.float64, 
                                    device=args.device)
        self.t_steps = (self.sigma_max ** (1 / args.rho) + \
                   step_indices / (args.n_steps - 1) * \
                   (self.sigma_min ** (1 / args.rho) - \
                    self.sigma_max ** (1 / args.rho))) ** args.rho
        self.t_steps = torch.cat([self.net.round_sigma(self.t_steps), 
                                  torch.zeros_like(self.t_steps[:1])])
               
        #Network error: subset of steps
        if args.subsample_Ct is not None:
            self.subsample_sigma(args)
            
        else:
            self.t_Csteps = self.t_steps
            
            
        #base name of the denoiser
        name_t = 'subs_' if len(self.t_Csteps) < len(self.t_steps) else ''
        self.denoiser_C_f = args.denoiser_C_f + '_' + name_t +\
              f'{self.rho}_{self.sigma_max}_{self.sigma_min}'

        #__________DATA TRANSFORMS__________________________________________________
               #General transforms.
        transforms_list = []
               #rescaling
        if args.mean_std: 
            assert args.min_max is None
            self.mean_std = torch.tensor(args.mean_std).to(self.device)
            transforms_list.append(lambda x: rescale_meanstd(x, self.mean_std))

        elif args.min_max:
            assert args.mean_std is None
            self.min_max = torch.tensor(args.min_max).to(self.device)
            transforms_list.append(lambda x: rescale_minmax(x, self.min_max))

            #crop the 0-padded area after UNet forward pass if nexcessary
        if (self.padded_shp!=0).any(): 
            transforms_list.append(lambda x: crop_padded_area(x, self.padded_shp))
        
       
        if args.RPM == 'det_porores':
            from forward_operators import DetRPM_FaciestoLogRes as RPM #for now only deterministic RPM
            self.rpm_solver = RPM()
        else: self.rpm_solver = self.dummy 
        
        #__________HARD DATA_________________________________________________________
            #Hard data (HD) conditioning - direct observations
        if args.HD_cond:
            self.condtype.append('HD_cond')
            
            self.idx_p = args.HD_idx
            self.HD_dobs = torch.load(self.input_dir+args.HD_dobs, weights_only=False).to(self.device)
            self.HD_dobs += torch.randn_like(self.HD_dobs)*torch.tensor(args.HD_sigmaerr).to(self.device)
                
            self.HD_dobs[torch.isnan(self.HD_dobs)] = -9999
            self.HD_mask = torch.load(self.input_dir+args.HD_mask, weights_only=False).to(self.device)
            self.HD_sigmaerr = args.HD_sigmaerr
            
            self.fwd_solver = lambda x: x[0,args.HD_idx] #just select where the model is being conditioned
            self.HD_transf = Compose_datat(transforms_list) #conditioning on direct observations should have not additional transformations
            
            #get or calculate denoiser uncertainty
            recompute_Cinv = True
            self.denoiser_C_fn = self.denoiser_C_f+'.npy'
            try: 
                recompute_Cinv = False
                self.cov_xhat0 = np.load(self.input_dir+'/'+self.denoiser_C_fn, mmap_mode='r+')
            except: 
                self.cov_xhat0 = compute_Cxhat0(args, self)
            
            #pre-compute the inverse covariance
            self.invCtot_f = args.invCtot_f + '_' + name_t +\
                   f'{self.rho}_{self.sigma_max}_{self.sigma_min}_HDerr{self.HD_sigmaerr}'+'.npy'
            
            #if there is not precomputed, compute full covariance inverse and save the memmap
            if recompute_Cinv:
                self.HD_invCov = prec_inverse(self, args)
            else:
                try: self.HD_invCov = np.load(self.save_dir+'/'+self.invCtot_f, mmap_mode='r+')
                
                except: 
                    self.HD_invCov = prec_inverse(self, args)
                    
            self.ndim = self.HD_mask.sum().item()
            
        #__________GEOPHYSICS____________________________________________________________
        elif args.PH_cond == 'ERT':
            #CDPS.dobs_pygimli
            #CDPS.elec_
            
            #------------------------------------------------------------
            # transformations if constrained on physics
            # define forward
            # load observed and data covariance 
            # full covariance is computed per iteration so no precomputing is possible
            #------------------------------------------------------------
            pass
        
        elif args.PH_cond == 'Seismic':
            raise NotImplementedError

        #__________ERT RE-INTERPRETATION (Downsampling)______________________________
        elif args.DS_cond:
            from forward_operators import MRM

            self.condtype.append('DS_cond')
            self.idx_p = args.DS_idx #this should go into the forward solver
            
            transforms_list.append(lambda x: flip(x)) #to match pygimli MCM MRM and m_est
            self.FW_transf = Compose_datat(transforms_list)
            
            self.denoiser_C_fn = self.denoiser_C_f+'_DS.npy'
            try: self.cov_xhat0 = np.load(self.input_dir+'/'+self.denoiser_C_fn, mmap_mode='r+')
            except: self.cov_xhat0 = compute_Cxhat0(args, self)
            
            self.fwd_solver = MRM(self, args.MRM_f)
            
            #take posterior covariance of the data
            self.C_dobs = torch.load('./'+self.input_dir+'/'+args.MCM_f,
                                     weights_only=True).to(self.device)
            
            #take mest as observed data
            self.FW_dobs = torch.load('./'+self.input_dir+'/'+args.mest_f,
                                     weights_only=True).to(self.device).flatten()
            
            
            #precompute the inverse covariance matrix for a subset of sampling steps
            self.invCtot_f = args.invCtot_f + '_' + name_t +\
                   f'{self.rho}_{self.sigma_max}_{self.sigma_min}'+'.npy'
    
            #if there is not precomputed, compute full covariance inverse and save the memmap
            try: self.FW_invCov = np.load(self.save_dir+'/'+self.invCtot_f, mmap_mode='r+')
            except: self.FW_invCov = prec_inverse(self, args)
            
            self.ndim = np.prod(self.shp[-2:])
        
        #for now all the inverse covariance matrices are pre-computed so this is needed in all the cases
        self.invCov = lambda x,y: torch.from_numpy(interp_ctot(self, i=x, invcov=y)).to(self.device).double().detach()

        self.HD_cond=args.HD_cond; self.DS_cond=args.DS_cond; self.PH_cond=args.PH_cond
        

    def dummy(self,x):
        return x
    
    
    def subsample_sigma(self, args):
        # subsample_sigma
        idxs = (((self.n_steps))*np.array(args.subsample_Ct)).round()-1; idxs[0] = 0 #get indexes given the proportions
        idxs = np.append(idxs,self.n_steps) #include sigma_0
        self.t_Csteps = self.t_steps[idxs]
        
        return None


    def full_covariance(self, cov_xhat0, operator = None, p=0):
        if self.condtype[-1] != 'HD_cond':
            if operator is not None: Cm0 = operator @ cov_xhat0 @ operator.T #check how for ERT and seismic
            else: Cm0 = self.fwd_solver.operator @ cov_xhat0 @ self.fwd_solver.operator.T
            C_tot = Cm0 + self.C_dobs
        else: C_tot = torch.eye(len(cov_xhat0.diag())).to(self.device)*self.HD_sigmaerr[self.idx_p[p]]**2 + cov_xhat0
        return C_tot


    def score_HD(self, xhat0, x_t, i):
        """Calculates gradients for hard data"""
        x_1 = self.HD_transf(xhat0)
        x = self.fwd_solver(self.rpm_solver(x_1))
        diff = ((self.HD_dobs - x) * self.HD_mask).flatten()
        L = -.5*(diff @ self.invCov(i, self.HD_invCov) @ diff)
        grads = torch.autograd.grad(outputs=L, inputs=x_t, retain_graph=False)[0].detach()
        
        diff = ((self.HD_dobs - x) * self.HD_mask).flatten().detach().cpu().numpy()
        DCD = (diff @ self.HD_invCov[-1] @ diff)
        wrmse = np.sqrt(DCD/self.ndim)
        
        return grads, wrmse


    def score_FwOp(self, xhat0, x_t, i):
        """Calculates gradients for geophysics / tomography"""
        x = self.FW_transf(xhat0)
        diff = self.FW_dobs - self.fwd_solver(self.rpm_solver(x[:,self.idx_p])[0])
        L = -.5*(diff @ self.invCov(i, self.FW_invCov) @ diff)
        grads = torch.autograd.grad(outputs=L, inputs=x_t, retain_graph=False)[0].detach()
        
        diff = diff.detach().cpu().numpy()
        DCD = (diff @ self.FW_invCov[-1] @ diff)
        wrmse = np.sqrt(DCD/self.ndim)

        return grads, wrmse


    #this and its initialization will become a class to account for other types of probabilistic models
    def EDM_sampling(self, latents = None, class_labels = None, 
                      randn_like=torch.randn_like, 
                      S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        
        if latents is None: latents = torch.randn(self.minshp).to(self.device)
        
        wrmse_tot = torch.zeros(self.n_steps, np.sum([self.HD_cond, self.DS_cond, self.PH_cond]))
        
        # Main sampling loop.
        desc = f'Condintioning - {self.minibatch} samples, {self.n_steps} steps'
        x_next = latents.to(torch.float64) * self.t_steps[0]
        
        pbar = tqdm(enumerate(zip(self.t_steps[:-1], self.t_steps[1:])), desc= desc)
        for i, (t_cur, t_next) in pbar: # 0, ..., N-1
            torch.cuda.empty_cache()
            x_cur = x_next
            
            # Increase noise temporarily.
            gamma = min(S_churn / self.n_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            
            #evaluate
            x_hat.requires_grad_(True)
            denoised = self.net(x_hat, t_hat, class_labels) #get xhat_0|x_t
            if self.has_lbl: 
                label = denoised[1]
                denoised = denoised[0]

            if self.HD_cond: cond_grads, wrmseHD = self.score_HD(denoised, x_hat, i) #get HD gradients dL/dx_t
            if self.DS_cond or self.PH_cond: cond_grads, wrmsePH = self.score_FwOp(denoised, x_hat, i) #get HD gradients dL/dx_t
            
            #preserve memory
            denoised.detach_(); x_hat.detach_(); x_next.detach_(); torch.cuda.empty_cache()
            
            #compute next step (Euler)
            score_func = (denoised-x_hat)/(t_hat**2)
            d_cur = - t_hat * (score_func + cond_grads)
            
            x_next = x_hat + (t_next - t_hat) * d_cur
           
            # Apply 2nd order correction.
            if i < self.n_steps - 1 and self.sec_ord:
                # evaluate
                x_next.requires_grad_(True)
                denoised = self.net(x_next, t_next.unsqueeze(0), class_labels) #get xhat_0|x_next
                if self.has_lbl: 
                    label = denoised[1]
                    denoised = denoised[0]
                
                if self.HD_cond: cond_grads, wrmseHD = self.score_HD(denoised, x_next, i+1) #get HD gradients dL/dx_t
                if self.DS_cond or self.PH_cond: cond_grads, wrmsePH = self.score_FwOp(denoised, x_next, i+1) #get HD gradients dL/dx_t
                
                #preserve memory
                denoised.detach_(); x_hat.detach_(); x_next.detach_(); torch.cuda.empty_cache()
                
                #compute next step (2nd order)
                score_func = (denoised-x_next)/(t_next**2)
                d_prime = - t_next * (score_func + cond_grads)
                
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
            if self.HD_cond: wrmse_tot[i][0] = wrmseHD[0]
            else: wrmse_tot[i][0] = wrmsePH[0]
            
            pbar.set_postfix({'WRMSE': wrmse_tot[i]}, refresh=False)

        if self.has_lbl: return x_next.detach().cpu(), wrmse_tot, label
        else: return x_next.detach().cpu(), wrmse_tot
                

    def __call__(self):

        results = torch.zeros(self.shp)
        labels = []
        wrmses = torch.zeros(self.n_samples,self.n_steps)
        
        for i in range(self.n_samples):
            sample = self.EDM_sampling()
            results[i] = sample[0]
            wrmses[i] = sample[1].squeeze()
            if self.has_lbl : 
                labels.append(sample[2])
                torch.save(labels, self.save_dir+'/labels.pt')
            
            torch.save(results, self.save_dir+'/results.pt')
            torch.save(wrmses, self.save_dir+'/wrmses.pt')
        
        results = flip(self.FW_transf(results.to(self.device)).detach().cpu())
        torch.save(results, self.save_dir+'/results.pt')
        if self.has_lbl: 
            return results, wrmses, labels
        else:
            return results, wrmses


