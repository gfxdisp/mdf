import torch
import torch.nn as nn
#from DQ import DQ
#from Loss_Modules.VGGPerceptualLoss import VGGPerceptualLoss
#from VGG_Normalized import VGG_Normalized
#from Loss_Modules.SpatialGradientLoss import SpatialGradientLoss
import torch.nn.functional as F
import kornia
import kornia.color as kcol

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    dtype = torch.FloatTensor
    torch.set_default_tensor_type('torch.FloatTensor')
    
class Multi_Scale(nn.Module):
    """Multi Scale class package for running different loss function on an image pyramid
    Arguments:
    1) Loss_function -> The nn.Module loss function you want to implement at different scales -> For example: nn.MSELoss(), DQ(), etc. Default value: DQ()
    2) num_levels_in_an_octave -> Number of levels desirable in the image pyramid. At each level, the image dimensions are half of the previous level
    3) Starting_sigma -> The starting std deviation of the Gaussian kernel that is used for incremental blur of the images
    4) pyramid_type -> Whether the loss function should be taken on a Gaussian or a Laplacian (i.e. difference of Gaussians) pyramid: Values -> 'Gaussian', 'Laplacian', 'Both'
    5) Lam -> If pyramid_type = 'Both', select the weight of the laplacian pyramid to be taken. The loss function will thus be: loss = gaussian_loss + lam * laplacian_loss
    6) Enhanced_Penalty (experimental) -> Applies a weightage to the loss such that loss = loss * exp(alpha*loss). This is based on an intuition that the loss will converge faster, but this feature is yet to be tested
    7) If enhanced penalty is chosen, then determine the alpha weight in the function loss = loss * exp(alpha*loss).
    Return:
        loss -> The corresponding loss function

    Shape:
        - Input: : PyTorch Tensor of shape (B, C, H, W)
        - Output: : loss between test and reference image
        
    """
    def __init__(self,loss_func=nn.MSELoss(), num_levels_in_an_octave = 5, starting_sigma = 0.6, pyramid_type = 'Gaussian',enhanced_penalty=False,lam=0.1,alpha=10,use_cuda=True):
        super(Multi_Scale,self).__init__()
        self.use_cuda = use_cuda
        self.num_levels_in_an_octave = num_levels_in_an_octave
        self.starting_sigma = starting_sigma
        self.pyramid_type = pyramid_type
        self.lam = lam
        self.enhanced_penalty = enhanced_penalty
        self.alpha = alpha
        self.loss_func=loss_func
    def forward(self,inp,target):
        sp_inp, sigmas, pds = kornia.ScalePyramid(self.num_levels_in_an_octave, self.starting_sigma)(inp)
        sp_target, sigmas, pds = kornia.ScalePyramid(self.num_levels_in_an_octave, self.starting_sigma)(target)
        
        # Shape of sp_inp tensor at a octave i = B*N*C*W*H where N = number of levels in an octave
    #     count = 0
        loss_func=self.loss_func
        try:
            loss_func.use_cuda=self.use_cuda
        except:
            print("Module does not contain manual use_cuda attribute. Continuing with the assumption that input and target tensors are loaded on same device")
        loss_taylor_l = torch.Tensor([1])
        if self.pyramid_type == 'Laplacian' or self.pyramid_type == 'Both':
            for i in range(len(sp_inp)):
                for j in range(self.num_levels_in_an_octave-1):
                    diff_inp_j = sp_inp[i][:,j+1,:,:,:] - sp_inp[i][:,j,:,:,:]
                    diff_target_j = sp_target[i][:,j+1,:,:,:] - sp_target[i][:,j,:,:,:]
                    if not torch.isnan(diff_inp_j).any() and not torch.isnan(diff_target_j).any():
                        loss_ij = loss_func(diff_inp_j,diff_target_j)
                        if not torch.isnan(loss_ij).any():
                            loss_inv = torch.abs((1-loss_ij))
                            # Taking taylor series expansion of loss_inv^0.15 (I empirically tested this to be a good hyperparameter)
                            loss_taylor_l *= 1+(3/20)*(loss_inv-1)-(51/800)*((loss_inv-1)**2)+(629/16000)*((loss_inv-1)**3)-(35853/1280000)*((loss_inv-1)**4)
            final_loss_l = 1-loss_taylor_l
        
        loss_taylor_g = torch.Tensor([1])
        if self.pyramid_type == 'Gaussian' or self.pyramid_type == 'Both':
            for i in range(len(sp_inp)):
                for j in range(self.num_levels_in_an_octave):
                    inp = sp_inp[i][:,j,:,:,:]
                    tgt = sp_target[i][:,j,:,:,:]
                    if not torch.isnan(inp).any() and not torch.isnan(tgt).any():
                        loss_ij = loss_func(inp,tgt)
                        if not torch.isnan(loss_ij).any():
                            loss_taylor_g  *= torch.abs((1-loss_ij))**0.15
            final_loss_g = 1-loss_taylor_g
        if self.pyramid_type == 'Both':
            loss = (1-self.lam)*final_loss_g  + self.lam*final_loss_l
            if self.enhanced_penalty:
                loss = loss*torch.exp(self.alpha*loss)
            return loss
        elif self.pyramid_type == 'Gaussian':
            if self.enhanced_penalty:
                final_loss_g = final_loss_g*torch.exp(self.alpha*final_loss_g)
            return final_loss_g
        elif self.pyramid_type == 'Laplacian':
            if self.enhanced_penalty:
                final_loss_l = final_loss_l*torch.exp(self.alpha*final_loss_l)
            return final_loss_l
