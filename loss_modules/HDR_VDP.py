import torch.nn as nn
import torch
import torch.nn.functional as F
import kornia
import math
import numpy as np
import operator
from collections.abc import Sequence
import contextlib
from scipy.interpolate import interp1d
from matplotlib.image import imread
from scipy import signal
import scipy.io
import matplotlib.pyplot as plt
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from bicubic import bicubic
bicubic = bicubic()

#### THIS IS A DIFFERENTIABLE VERSION OF HDR VDP THAT CAN BE INCORPORATED AS A VISUAL LOSS FUNCTION IN NEURAL NETWORKS #####
#### FEW COMMENTS WHICH MIGHT BE HELPFUL IN DEBUGGING: ####
#### 1) Laplacian pyramid results are the same as Matlab (gausspyr reduce and expand are working properly)
#### 2) Mutual Masking is working
#### 3) get_luminance and hdrvdp_gog_display_model is working
#### 4) Error is occuring somewhere between lines 430-450 -> Probably need to check gaussian filtering again
####


def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape

def gausspyr_reduce(x,kernel_a=0.4):
    K = torch.Tensor([ 0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2 ])
    K =  K.expand(x.shape[1],1,K.shape[0]).unsqueeze(-1)
    x = F.pad(x, ((K.shape[3])//2,(K.shape[3])//2,(K.shape[2])//2,(K.shape[2])//2))
    y_a = torch.transpose(F.conv2d(x,K,groups = x.shape[1]),2,3)
    y = y_a[:,:,:,0:y_a.shape[3]:2]
    y = F.pad(y, ((K.shape[3])//2,(K.shape[3])//2,(K.shape[2])//2,(K.shape[2])//2))
    y_a = F.conv2d(y,K,groups = y.shape[1])
    y = y_a[:,:,0:y_a.shape[2]:2,:]
    return y

def gausspyr_expand(x,sz=None,kernel_a=0.4):
    if sz is None:
        sz = (x.shape[2]*2,x.shape[3]*2);
#     print("Desired size: " + str(sz))
    m1 = sz[1]%2
    ch_no = x.shape[1]
    K = 2*torch.Tensor([ 0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2 ])
    K =  K.expand(x.shape[1],1,K.shape[0]).unsqueeze(-1)
    y_a = torch.zeros((x.shape[0],ch_no,x.shape[2],sz[1]+m1))
    y_a[:,:,:,:-1:2] = x
    y_a = F.pad(y_a, ((K.shape[2])//2,(K.shape[2])//2,(K.shape[3])//2,(K.shape[3])//2))
    y_a = F.conv2d(y_a,torch.transpose(K,2,3),groups = y_a.shape[1])
    m0 = sz[0]%2
    y = torch.zeros((x.shape[0],ch_no,sz[0]+m0,sz[1]+m1))
    
    y[:,:,:-1:2,:] = y_a
    
    y = F.pad(y, ((K.shape[3])//2,(K.shape[3])//2,(K.shape[2])//2,(K.shape[2])//2))
    y = F.conv2d(y,K,groups = y.shape[1])
#     print(y,y.shape)
#     print("Expanded pyramid mean: " + str(y[:,:,:sz[0],:sz[1]].mean()))
    return y[:,:,:sz[0],:sz[1]]


def gaussian_window(window_size,sigma,num_channels):
    window = torch.Tensor(signal.gaussian(window_size, std=sigma)).unsqueeze(-1)
    # Because the gaussian is a separable filter, we can do torch.mm on the window to get the 2D filter
    window = window / window.sum()
    window_2d = window.mm(window.t())
    window_2d = window_2d.expand(num_channels,1,window_size,window_size)
    # need to output same channels as image while convolving
    # -> torch.nn.functional documentation wants weights in the format (output_size, input_size/groups,h,w)
    return window_2d


class gaussian_pyramid_dec(nn.Module): ## Generates a gaussian kernel with the given parameters
    """ Image is assumed to be of shape B*C*H*W """
    def __init__(self,levels=None,kernel_a=None):
        self.levels = levels
        self.kernel_a = kernel_a
        
    def forward(self,img):
        default_levels = torch.floor(torch.log2(torch.min(torch.Tensor([img.shape[2]]),torch.Tensor([img.shape[3]]))))
        if self.levels is None:
            self.kernel_a = 0.4
            self.levels = -1
        
        if self.kernel_a is None:
            self.kernel_a = 0.4
            
        if self.levels == -1:
            self.levels = default_levels;
        
        if self.levels > default_levels:
            print(self.levels)
            raise ValueError("Parameter 'levels' too large!" )
        leveldict = {}
        levels = int(self.levels)
        leveldict[1] = img
        
        if int(levels) == 1:
            return leveldict
        for i in range(2,int(levels)+1):
            leveldict[i] = gausspyr_reduce(leveldict[i-1], self.kernel_a);
            
        if len(leveldict[levels]) == 0:
            leveldict={};
        return leveldict


class laplacian_pyramid_dec(nn.Module):## Generates a decimated laplacian pyramid of the image
    def __init__(self,levels=None,kernel_a=None):
        super(laplacian_pyramid_dec,self).__init__()
        self.levels = levels
        self.kernel_a = kernel_a
        if self.levels is None:
            self.levels = -1
        
        if self.kernel_a is None:
            self.kernel_a = 0.4

    def forward(self,img):
        g_pyramid=gaussian_pyramid_dec(self.levels,self.kernel_a)
        g_pyramid = g_pyramid.forward(img)
        height = len(g_pyramid)
        if height == 0:
            return {}
        lpyr = {}
        for i in range(1,height):
            try:
                e = gausspyr_expand(g_pyramid[i+1],(g_pyramid[i].shape[2], g_pyramid[i].shape[3]), self.kernel_a)
            except:
                e = gausspyr_expand(torch.transpose(g_pyramid[i+1],2,3),(g_pyramid[i].shape[2], g_pyramid[i].shape[3]), self.kernel_a)
            lpyr[i] = g_pyramid[i] - e
        lpyr[height] = g_pyramid[height]
        return lpyr
        
        
    
class hdrvdp_lpyr_dec(nn.Module):# Wrapper class for implementing decimated laplacian pyramid structure on an image
    def __init__(self,P=None,ppd=None,base_ppd=None,img_sz=None,band_freqs=None,height=None):
        super(hdrvdp_lpyr_dec,self).__init__()
        self.P = P
        self.ppd = ppd
        self.base_ppd = base_ppd
        self.img_sz = img_sz
        self.band_freqs = band_freqs
        self.height = height
        
    def decompose(self,I,ppd):
        self.ppd = ppd;
        self.img_sz = I.shape;
        self.height = torch.max(torch.ceil(torch.log2(torch.Tensor([ppd]))),torch.Tensor([1]))
#         print(self.height)
        x = torch.Tensor([i for i in range(0,int(self.height))])
        y = 0.3228*(2**(-x))
        z = torch.Tensor([1])
        self.band_freqs = torch.cat([z,y])*ppd/2
#                   % We want the minimum frequency the band of 0.5cpd or higher
#       % Corrected frequency peaks of each band
        lap_pyr = laplacian_pyramid_dec(levels=self.height+1) # height + 1
        self.P = lap_pyr(I)
        
        return self.P
        
    def reconstruct(self):
        I = self.P[len(self.P)]
        
        for i in range(len(self.P)-1,0,-1):
            I = F.interpolate(I,size=(self.P[i].shape[2],self.P[i].shape[3]),mode='bicubic')
            I += self.P[i]
        return I
    
    def get_band(self,band,o):
        if band == 1 or band == len(self.P):
            band_mult = 1;
        else:
            band_mult = 2;
            
        B = self.P[band]* band_mult;
        return B
    
    def set_band(self, band, o,B):
        
        if band == 1 or band == len(self.P):
            band_mult = 1;
        else:
            band_mult = 2;
        
        self.P[band] = B/band_mult;
        return
        
    def band_count(self):
        return len(self.P);
    
    def orient_count(self, band):
        oc = 1;
        return oc
    
    def band_size(self,band,o):
        return self.P[band].shape
        
    def get_freqs(self):
        return self.band_freqs
        

class HDR_VDP(nn.Module):
    ''' Main HDR VDP Loss Function Module. Expects input tensors to be of shape B*C*H*W
        Example usage: x = torch.rand(128,3,250,250)
                       y = torch.rand(128,3,250,250)
                       hdr_vdp = HDR_VDP()
                       loss = hdr_vdp(x,y)
        In this case, x is the model output and y is the reference image
    '''
    def __init__(self,resolution=[3840,2160],display_size_m=None,display_size_deg=None,distance_m=None,fixed_ppd=None):
        super(HDR_VDP,self).__init__()
        self.Y_peak = 100
        self.contrast = 1000
        self.gamma = 2.2
        self.E_ambient = 0
        self.resolution = resolution
        self.display_size_m = display_size_m
        self.display_size_deg = display_size_deg;
        self.distance_m = distance_m;
        self.fixed_ppd = fixed_ppd
        self.k_mask_self = 1;
        self.mask_p = 2.2;
        self.mask_q = 2.4;
        self.pu_dilate = 3;
        self.debug = False;
        self.beta = 3.5
        self.csf_sigma = -1.06931;
        self.do_foveated = False
        self.B = {}
    
    def get_luminance(self,img):
        """ Return 2D matrix of luminance values for 3D matrix with an RGB image. Image shape = (N*C*W*H)"""
        
        if len(img.shape) > 4:
            raise ValueError("Wrong matrix dimension!")
        if img.shape[1] == 1:
            Y = img
        elif img.shape[1] == 3:
            Y = img[:,0,:,:] * 0.212656 + img[:,1,:,:] * 0.715158 + img[:,2,:,:] * 0.072186;
        return Y.unsqueeze(1)
        
    def hdrvdp_display_resolution(self,resolution=[3840,2160],distance_display_heights=None,fov_horizontal=None,fov_vertical=None, fov_diagonal=None,display_diagonal_in = 30,distance_m=0.5):
        if len(resolution) == 1:
            self.fixed_ppd = resolution
        else:
            pass
        aspect_ratio = resolution[0]/resolution[1];
        
        if display_diagonal_in is not None:
            height_mm = math.sqrt((display_diagonal_in*25.4)**2/(1+aspect_ratio**2))
            self.display_size_m = []
            self.display_size_m.append(aspect_ratio*height_mm/1000)
            self.display_size_m.append(height_mm/1000)
        if distance_m is not None:
            self.distance_m = distance_m
        elif distance_display_heights is not None:
            if self.display_size_m is None:
                raise ValueError("You need to specify display diagonal size ''display_diagonal_in'' to specify viwing distance as ''distance_display_heights'' ")
            self.distance_m = distance_display_heights * self.display_size_m[1]
        else:
            raise ValueError("Viewing distance must be specified as ''distance_m'' or ''distance_display_heights''");
        if fov_horizontal is not None:
            width_m = 2*math.tan( (fov_horizontal/2)*math.pi/180 )*self.distance_m;
            self.display_size_m = []
            self.display_size_m.append(width_m)
            self.display_size_m.append(width_m/aspect_ratio)
        elif fov_vertical is not None:
            height_m = 2*math.tan((fov_vertical/2)*math.pi/180 )*self.distance_m;
            self.display_size_m = []
            self.display_size_m.append(height_m*aspect_ratio)
            self.display_size_m.append(height_m)
        elif fov_diagonal is not None:
            distance_px = sqrt(sum([i**2 for i in resolution])) / (2.0 * math.tan(fov_diagonal * 0.5*math.pi/180));
            height_deg = math.atan(resolution[1]/2/distance_px)*360/math.pi
            height_m = 2*math.tan(math.pi*height_deg/360 )*self.distance_m;
            self.display_size_m = []
            self.display_size_m.append(height_m*aspect_ratio)
            self.display_size_m.append(height_m)
        self.display_size_deg = 2*np.arctan(np.array(self.display_size_m)/(2*self.distance_m))*180/math.pi
        return
        
    def get_ppd(self,eccentricity=None):
        if self.fixed_ppd is not None:
            ppd = self.fixed_ppd
            return ppd
        pix_deg = 2*math.atan( 0.5*self.display_size_m[0]/self.resolution[0]/self.distance_m )*180/math.pi
        base_ppd = 1/pix_deg
        if eccentricity is None:
            ppd = base_ppd
        else:
            delta = pix_deg/2
            tan_delta = math.tan(delta*math.pi/180)
            tan_a = math.atan(eccentricity)*180/math.pi
            ppd = base_ppd*math.tan(((eccentricity+delta)-tan_a)*pi/180)/tan_delta
        self.fixed_ppd = ppd
        return ppd
    
    def get_resolution_magnification(self,eccentricity):
        if self.fixed_ppd is not None:
            M = 1
            shape = get_shape(eccentricity)
            for i in shape:
                M = [M]*i
            return M
        
        pix_deg = 2*math.atan( 0.5*self.display_size_m[0]/self.resolution[0]/self.distance_m )*180/math.pi
        delta = pix_deg/2
        tan_delta = math.tan(delta*math.pi/180)
        tan_a = math.atan(eccentricity)*180/math.pi
        M = math.tan(((eccentricity+delta)-tan_a)*pi/180)/tan_delta
        return M
        
    def hdrvdp_gog_display_model(self,V, Y_peak, contrast,gamma, E_ambient, k_refl=None):
        if contrast is None:
            contrast = 1000
        if gamma is None:
            gamma = 2.2
        if E_ambient is None:
            E_ambient = 0
        if k_refl is None:
            k_refl = 0.005
        Y_refl = E_ambient/math.pi*k_refl
        
        Y_black = Y_refl + Y_peak/contrast
        Y = (Y_peak - Y_black)*(torch.pow(V,gamma))+Y_black
        return Y
    
    def mutual_masking(self,b,o):
        pix_per_deg = self.fixed_ppd
        test_band = self.B[1].get_band(b,o);
        reference_band = self.B[2].get_band(b,o);
        m = torch.min(torch.abs(test_band),torch.abs(reference_band))
        if len(test_band.shape) == 2:
            test_band = test_band.unsqueeze(0).unsqueeze(0)
            reference_band = reference_band.unsqueeze(0).unsqueeze(0)
        elif len(test_band.shape) == 3:
            test_band = test_band.unsqueeze(0)
            reference_band = reference_band.unsqueeze(0)
            
        if self.pu_dilate != 0:
            window = gaussian_window(window_size=2*math.ceil(2*self.pu_dilate)+1,sigma=self.pu_dilate,num_channels=test_band.shape[1])
            m = F.pad(m,((window.shape[3])//2,(window.shape[3])//2,(window.shape[2])//2,(window.shape[2])//2), mode='replicate')
            m = F.conv2d(m,window,groups = m.shape[1])
            
        return m
        
    def minkowski_sum(self,X,p):
        ''' X is of shape B*C*H*W '''
        numel = float(X.shape[1]*X.shape[2]*X.shape[3])
        d = torch.sum((torch.abs(X)**p)/numel,dim=(1,2,3))**(1/p)
        return d # Returns a vector of length batch size with corresponding minkowski sums

    def forward(self,T_img,R_img,**kwargs):
        self.hdrvdp_display_resolution(resolution=[3840,2160],display_diagonal_in= 30, distance_m = 0.5)
        ppd = self.get_ppd()
        pix_per_deg = ppd
        if kwargs is None:
            kwargs = {}
        attributes = {'k_mask_self':1, 'mask_p': 2.2, 'mask_q': 2.4, 'pu_dilate': 3, 'debug': False, 'beta': 3.5, 'csf_sigma':-1.06931, 'do_foveated': False,}
        for key, value in kwargs.items():
            if key not in attributes:
                raise ValueError("Unknown option {}".format(key))
            attributes[key] = value;
        T_img = self.get_luminance(T_img)
        R_img = self.get_luminance(R_img)
        T_img = self.hdrvdp_gog_display_model(T_img, self.Y_peak, self.contrast, self.gamma, self.E_ambient);
        R_img = self.hdrvdp_gog_display_model(R_img, self.Y_peak, self.contrast, self.gamma, self.E_ambient);
        Y_min = 1e-6; # Minimum allowed values (avoids NaNs)
        T_img[T_img < Y_min] = Y_min
        R_img[R_img < Y_min] = Y_min
        T_img = T_img.cuda()
        R_img = R_img.cuda()
        
        window = gaussian_window(2*math.ceil(2*0.5*pix_per_deg)+1,0.5*pix_per_deg,T_img.shape[1])
        x = F.pad(R_img,((window.shape[3])//2,(window.shape[3])//2,(window.shape[2])//2,(window.shape[2])//2),mode='replicate')
        L_adapt = F.conv2d(x,window,groups=R_img.shape[1])
#         print(L_adapt)
#        L_adapt = kornia.filters.GaussianBlur(R_img[:,0:1,:,:],sigma=0.5*pix_per_deg, kernel_size=2*math.ceil(2*0.5*pix_per_deg)+1)
        ms_ref = hdrvdp_lpyr_dec();
        ms_test = hdrvdp_lpyr_dec();
        N_nCSF = [];
        decomposed_ref = ms_ref.decompose(R_img,ppd)
        decomposed_test = ms_test.decompose(T_img,ppd)
#         print(decomposed_ref)
#         pyrx = decomposed_ref
        self.B[1] = ms_test
        self.B[2] = ms_ref
        rho_band = ms_test.get_freqs()
        nCSF_bins = 64;
        l_nCSF = torch.Tensor(np.linspace( np.log10(0.001), np.log10(10000), nCSF_bins));
        for bb in range(1,ms_test.band_count()):
            if self.csf_sigma < 0:
                sigma_f = -self.csf_sigma / rho_band[bb]
            else:
                sigma_f = self.csf_sigma
            A = math.pi*sigma_f**2
        N_nCSF = torch.Tensor(np.loadtxt(open("N_nCSF.csv", "rb"), delimiter=","))
#         print(N_nCSF)
        
        for bb in range(1,ms_test.band_count()):
            T_f = self.B[1].get_band(bb,1);
            R_f = self.B[2].get_band(bb,1);
#             print(T_f.mean())
            L_bkg = F.interpolate(L_adapt,size=(T_f.shape[2],T_f.shape[3]),mode='area')
            L_bkg[L_bkg < 1e-4] = 1e-4
            max_contrast = 1000
            D_ex = torch.abs((T_f - R_f)/L_bkg)
            D_ex[D_ex > max_contrast] = max_contrast
            D_ex =torch.pow(D_ex,self.mask_p)
            self_mask = self.mutual_masking(bb,1)/L_bkg
            N_mask = (self.k_mask_self*torch.abs(self_mask))**self.mask_q;
            L_bkg_limited = torch.clamp(torch.log10(L_bkg),l_nCSF[0], l_nCSF[-1]).cpu()
            L_bkg_limited = np.array(L_bkg_limited)
            N_static = 10**torch.Tensor(np.interp(L_bkg_limited,np.array(l_nCSF.cpu()),np.array(N_nCSF[:,bb-1].cpu())))
            D = D_ex / torch.sqrt(N_static**(2*self.mask_p)+N_mask**2)
            if bb == 1:
                Q_err = self.minkowski_sum(D, self.beta)/self.B[1].band_count();
            else:
                Q_err += self.minkowski_sum(D, self.beta)/self.B[1].band_count();
        return Q_err.mean()
   
hdr_vdp = HDR_VDP()

## x = torch.rand(64,3,250,250)
## y = torch.rand(x.shape)
## print(hdr_vdp(x,y))


