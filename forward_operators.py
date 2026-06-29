# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 11:28:29 2026
    phyiscs forward operators
    
    Options should include 
    - Seismic fullstack
    - ERT with pygimli (including init schemes etc)
    - Loading of matrix operators saved as files
    - Eventually: Seismic AVA (available from older codes) 
    
@author: rmiele
"""
import torch
import numpy as np
try: 
    import pygimli as pg
except:
    pass    

#________________________________________GEOPHYSICS______________________________________________________________
class MRM():
    """
        Class related to downsampling of tomography
    """
    def __init__(self, CDPS, filename):
        self.condtype = CDPS.condtype
        if 'DS_cond' in CDPS.condtype:
            self.operator = torch.load(CDPS.input_dir+'/'+filename, weights_only=True).to(CDPS.device)
            
    def __call__(self, array):
        if 'DS_cond' in self.condtype:
            array = array.type(self.operator.type())
            return self.operator @ array.squeeze().flatten()


class pg_fw_bk(torch.autograd.Function):    
    """
        Allows forward evaluation with pygimli 
        and computation of gradients with torch.autograd
    """
    @staticmethod
    def forward(ctx, x_hat, pygimli_obj):
        d_np = x_hat.new()
        # call C++ forward
        d_np = pygimli_obj.fop(x_hat.flatten().detach().cpu().numpy()).array()

        d_est = torch.from_numpy(d_np).to(x_hat.device) 
        
        pygimli_obj.fop.createJacobian(x_hat.flatten().detach().cpu().numpy())   # shape: [m, n]
        pygimli_obj.J = J = torch.tensor(pg.utils.gmat2numpy(pygimli_obj.fop.jacobian())).to(x_hat.device) 
        
        # save J for "manual" backward computation
        ctx.save_for_backward(J)
        ctx.shape = x_hat.shape

        return d_est

    @staticmethod
    def backward(ctx, grad_output):
        J, = ctx.saved_tensors
        grad_input = J.T @ grad_output
        
        return grad_input.reshape(ctx.shape)[None,None,:], None,None

    
class ERTPygimli():
    def __init__(self,CDPS):
        import pygimli.meshtools as mt
        from pygimli.physics import ert

        self.scheme = ert.createData(elecs = np.linspace(start=CDPS.elec_i, stop=CDPS.elec_f, 
                                                         num=int((CDPS.elec_f - CDPS.elec_i)/CDPS.spacing+1)),
                                     schemeName = CDPS.scheme) 
        
        halfd = CDPS.elec_f+CDPS.boundary*CDPS.spacing
        #create regular grid
        self.inversion_domain = pg.createGrid(x=np.linspace(start = -halfd, stop = halfd, 
                                                            num = halfd*2+1),
                                        y=np.linspace(start=-CDPS.shp[-2], stop=0, num=CDPS.shp[-2]+1),
                                        marker=2)
        #add buffer zone
        self.inv_grid = mt.appendTriangleBoundary(self.inversion_domain, marker=1,
                                                  xbound=CDPS.shp[-1]*3,
                                                  ybound=CDPS.shp[-2]*6,
                                     quality=32, smooth=True)

        self.mgr = ert.ERTManager(CDPS.dobs_pygimli)
        self.mgr.setMesh(self.inv_grid)

        return None
    
    def __call__(self,array):
        self.mgr.setData(self.scheme) #this idk if needed but some tests in the past suggested so
        
        return pg_fw_bk.apply(torch.exp(array), self.mgr)



class SeismicFullstack():
    def __init__(self, CDPS):
        raise NotImplementedError
        
    def load_wavelet_define_conv(self, args):
        # wavelet = np.genfromtxt(args.workdir+args.inv_folder+args.wavelet_file)*args.w_scale
        frequency = 1/(torch.pi*5)
        t= torch.arange(-20,20,1)
        omega = torch.pi * frequency
        wavelet = ((1 - 2 * (omega * t) ** 2) * torch.exp(-(omega * t) ** 2))*100
        
        wavelet = np.expand_dims(wavelet, 0) # add bacth [B x H x W x C]
        if wavelet.ndim==2: 
            wavelet= np.expand_dims(wavelet, 0)
            wavelet= np.expand_dims(wavelet, -1)
            
        self.wavelet = torch.from_numpy(wavelet).double().to(args.device)
        k = self.wavelet.shape[-2]
        self.padding = (k//2,0)
        self.seismic_conv = torch.nn.Conv2d(1,1, kernel_size=1, padding= self.padding, bias=False)
        self.seismic_conv.weight = torch.nn.Parameter(self.wavelet).requires_grad_(False)
        
        return None
    
    def physics_forward(self, realization):
        ip = torch.cat((realization, realization[:,:,[-1],:]), dim=2) # repeats last element
        ip_d =  ip[:, :, 1:, :] - ip[:, :, :-1, :]
        ip_a = (ip[:, :, 1:, :] + ip[:, :, :-1, :])    
        rc = ip_d / ip_a
        return self.seismic_conv(rc)[:,:,:self.image_size[0],:]


    def diag_els (self, xi, xii):
        #Jacobian of reflectivity coefficients  models
        den = (xi+xii)**2
        num1 = -2*xii
        num2 = 2*xi
        return torch.tensor([num1/den, num2/den])


    def conv_matrix_J(self):
        #creates the matrix for 1D convolution, input data is padded
        #same as torch, but torch is more efficient at forward, this is needed for Jacobian 
        wavelet_lenght = len(self.wavelet.squeeze())
        trace_lenght = self.image_size[0]+wavelet_lenght
        toeplitz_w = torch.zeros(trace_lenght-wavelet_lenght+1-wavelet_lenght%2, 
                                 trace_lenght) 
        
        for i in range(toeplitz_w.shape[0]):
            toeplitz_w[i,i:i+wavelet_lenght]=self.wavelet.squeeze()
            
        return toeplitz_w




#________________________________________ROCK PHYSICS______________________________________________________________
class DetRPM_FaciestoLogRes():
    def __init__(self, model_porosity=False):
        #m is the cementation exponent, found with a linear function dependent on the lithology
        #ranges considered are below
        
        #b0 is the slope; b1 is the intercept (found from linear regression of rock physics values - see notes)
        self.b0_poro = -0.063
        self.b1_poro = 0.36
        
        self.b0_m = -0.1
        self.b1_m = 2
        
        self.a = 1.1 #we just assume this one
        self.r_w = 20 #20 ohm m
        self.model_porosity= model_porosity
        
    def litho_to_poro(self,array):
        #computes porosity from lithology according to a linear function
        self.poro = array*self.b0_poro + self.b1_poro
        return self.poro
    
    def litho_to_m(self,array):
        #computes the cementation exponent from lithology according to a linear function
        self.m = array*self.b0_m + self.b1_m
        return self.m
    
    def fw_archies(self):
        #computes the resistivity values from instance cementation exponent and porosity (i.e., precomputed)
        self.res = self.r_w * (self.poro ** (- self.m ))
        return self.res
    
    def __call__(self, array):
        #full forward returns log resistivity
        if self.model_porosity: 
            self.litho_to_poro(array)
            self.litho_to_m(array)
            
        else:
            self.poro = array[:,1]
            self.litho_to_m(array[:,0])
        
        self.fw_archies()
        return self.res.log() 
    
