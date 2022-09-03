import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy import signal
import torch
import kornia

class DQ(nn.Module):
    def __init__(self,enhanced_penalty=False,multiple_moments=False,alpha=5,use_cuda=True):
        super(DQ,self).__init__()
        self.use_cuda=use_cuda
        self.enhanced_penalty = enhanced_penalty
        self.multiple_moments = multiple_moments
        self.alpha=alpha
        
    def gaussian_window(self,window_size,sigma,num_channels):
        window = torch.Tensor(signal.gaussian(window_size, std=sigma)).unsqueeze(-1)
        # Because the gaussian is a separable filter, we can do torch.mm on the window to get the 2D filter
        window = window / window.sum()
        window_2d = window.mm(window.t())
        window_2d = window_2d.expand(num_channels,1,window_size,window_size)
        # need to output same channels as image while convolving
        # -> torch.nn.functional documentation wants weights in the format (output_size, input_size/groups,h,w)
        if self.use_cuda:
            return window_2d.cuda()
        return window_2d
        
    def forward(self,inp,target):
        dq = 0
        window_size = 11
        if len(inp.shape) == 3: # 1st dimension is batch size
            inp = torch.reshape(inp,(-1,inp.shape[0],inp.shape[1],inp.shape[2]))
            target = torch.reshape(target,(-1,target.shape[0],target.shape[1],target.shape[2]))
        window = self.gaussian_window(window_size,1.5,inp.shape[1])
        # need padding to ensure output is the same size as shape
        mu_x = F.conv2d(inp,window,padding = 11//2,groups = inp.shape[1])
        mu_y = F.conv2d(target,window,padding = 11//2,groups = inp.shape[1])
        var_x = F.conv2d(inp*inp,window,padding = 11//2,groups = inp.shape[1]) - mu_x**2 # sigma^2 = E(x**2)-E(x)**2
        var_y = F.conv2d(target*target,window,padding = 11//2,groups = inp.shape[1])- mu_y**2
        cov_xy = F.conv2d(inp*target,window,padding = 11//2,groups = inp.shape[1]) - mu_y*mu_x
        if not self.multiple_moments:
            var_minus = var_x+var_y-2*cov_xy
            var_plus = var_x+var_y+2*cov_xy
            epsilon = 0.001
            DQ = (2*torch.abs(var_minus)+epsilon)/(torch.abs(var_plus+var_minus)+epsilon)
            DQ[DQ!=DQ] = 1
            if self.enhanced_penalty:
                mean_DQ = torch.mean(DQ)
                loss = mean_DQ*torch.exp(self.alpha*mean_DQ)
            else:
                loss = torch.mean(DQ)
            return loss
        else:
            epsilon = 0.001
            skew_x = F.conv2d(torch.pow(((inp-mu_x)/torch.sqrt(var_x+epsilon)),3),window,padding = 11//2,groups = inp.shape[1])
            skew_y = F.conv2d(torch.pow(((target-mu_y)/torch.sqrt(var_y+epsilon)),3),window,padding = 11//2,groups = inp.shape[1])
            skew_xxy = F.conv2d((inp-mu_x)*(inp-mu_x)*(target-mu_y),window,padding = 11//2,groups = inp.shape[1])/(torch.sqrt(torch.abs(var_x*var_x*var_y)+epsilon))
            skew_xyy = F.conv2d((inp-mu_x)*(target-mu_y)*(target-mu_y),window,padding = 11//2,groups = inp.shape[1])/(torch.sqrt(torch.abs(var_x*var_y*var_y)+epsilon))
            kurt_x = (F.conv2d((inp-mu_x)**4,window,padding = 11//2,groups = inp.shape[1])) / (var_x**2+epsilon)
            kurt_y = (F.conv2d((target-mu_y)**4,window,padding = 11//2,groups = inp.shape[1])) / (var_y**2+epsilon)
            kurt_xy = (F.conv2d(((inp-mu_x)**2)*((target-mu_y)**2),window,padding = 11//2,groups = inp.shape[1])) / (var_x*var_y+epsilon)
            c1 = 0.001
            c2 = 0.001
            c3 = 0.001
            c4 = 0.001
            #### For debugging purposes
            if torch.isnan(inp).any():
                print("Problem in inp")
            if torch.isnan(mu_x).any():
                print("Problem in mu_x")
            if torch.isnan(skew_xxy).any():
                print("Problem in skewxxy")
            if torch.isnan(skew_xyy).any():
                print("Problem in skew_xyy")
            if torch.isnan(skew_y).any():
                print("Problem in skew_y")
            if torch.isnan(skew_x).any():
                print("Problem in skew_x")
            kurt = (2*kurt_xy+c4)/(kurt_x+kurt_y+c4)
            if torch.isnan(kurt).any():
                print("Problem in kurt")
            skew = (2*skew_xxy*skew_xyy+c3)/(skew_x**2+skew_y**2+c3)
            if torch.isnan(skew).any():
                print("Problem in skew")
            var = (2*cov_xy+c2) / (var_x+var_y+c2)
            if torch.isnan(var).any():
                print("Problem in var")
            mean = (2*mu_x*mu_y+c1)/(mu_x**2+mu_y**2+c1)
            if torch.isnan(mean).any():
                print("Problem in mean")
            #### For debugging purposes
            kurt[kurt>1] = 1
            skew[skew>1] = 1
            SSIM_multi_moment = mean*var*skew*kurt
            SSIM_multi_moment[SSIM_multi_moment!=SSIM_multi_moment] = 0
            DQ = 1-SSIM_multi_moment
            if self.enhanced_penalty:
                mean_DQ = torch.mean(DQ)
                loss = mean_DQ*torch.exp(self.alpha*mean_DQ)
            else:
                loss = torch.mean(DQ)
            return loss

        
