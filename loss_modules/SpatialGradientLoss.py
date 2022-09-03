import torch.nn as nn
import kornia
import torch
import torch.nn.functional as F

def gaussian_window(window_size,sigma,num_channels):
    window = torch.Tensor(signal.gaussian(window_size, std=sigma)).unsqueeze(-1)
    # Because the gaussian is a separable filter, we can do torch.mm on the window to get the 2D filter
    window = window / window.sum()
    window_2d = window.mm(window.t())
    window_2d = window_2d.expand(num_channels,1,window_size,window_size)
    # need to output same channels as image while convolving
    # -> torch.nn.functional documentation wants weights in the format (output_size, input_size/groups,h,w)
    return window_2d.cuda()

class SpatialGradientLoss(nn.Module):
    """Computes the first order image derivative in both x and y using a Sobel
    operator.
    Arguments:
    
    1) Enhanced_Penalty (experimental) -> Applies a weightage to the loss such that loss = loss * exp(alpha*loss). This is based on an intuition that the loss will converge faster, but this feature is yet to be tested
    2) Alpha: If enhanced penalty is chosen, then determine the alpha weight in the function loss = loss * exp(alpha*loss).
    3) Orientation: If this is true, the loss will also be taken on the gradient orientations
    4) Lam -> By default, we apply small amounts of mean value regularisation. This parameter allows the amount of regularisation to be adjusted
    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: : (B, C, H, W)
        - Output: : (2, B, C, H, W)

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial gradient  # 2x1x3x4x4
    """

    def __init__(self,use_cuda=True,loss_func=nn.L1Loss(), orientation=False,lam=0.01):
        super(SpatialGradientLoss, self).__init__()
        self.use_cuda=use_cuda
        self.kernelx,self.kernely = self.get_sobel_kernel_3x3(self)
        self.orientation = orientation
#         self.loss_type=loss_type
        self.lam = lam
        self.loss_func = loss_func

    @staticmethod
    def get_sobel_kernel_3x3(self):
        """Utility function that returns a sobel kernel of 3x3"""
#         dx = -torch.tensor([[[[3, 0, -3], [10, 0,-10], [3,0,-3]]]])/16.0
#         dy = -torch.tensor([[[[3, 10, 3], [0, 0, 0],   [-3 ,-10, -3]]]])/16.0
        dx = -torch.tensor([[1,0,-1]])/1.0
#         dx = torch.FloatTensor([1,0,-1]).cuda()
        dy = dx.t()
        if self.use_cuda:
            return dx.cuda(),dy.cuda()
        return dx,dy
    def magnitude(self, input): # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, expected BxCxHxW. Got: {}".format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        grad = torch.zeros(2,b,c,h,w) # x and y
        grad[0,:,:,:,:] = F.conv2d(input,self.kernelx.expand(c,1,3,3),padding = 1,groups = c)
        grad[1,:,:,:,:] = F.conv2d(input,self.kernely.expand(c,1,3,3),padding = 1,groups = c)
        gradient = torch.sqrt(torch.pow(grad[0,:,:,:,:],2)+torch.pow(grad[1,:,:,:,:],2))
        # convolve input tensor with sobel kernel
#         kernel_flip = kernel.flip(-3)
        if torch.cuda.is_available():
            return gradient.cuda(),grad[0,:,:,:,:].cuda(),grad[1,:,:,:,:].cuda()
        return gradient,grad[0,:,:,:,:],grad[1,:,:,:,:]

    def orx(self,input):
        gradient, grad_x, grad_y = self.magnitude(input)
        epsilon = 0.001
        o = torch.atan((grad_y+epsilon)/(grad_x+epsilon))
        if torch.cuda.is_available():
            return o.cuda()
        return o
    
    def forward(self,x,y):
        grad_x,grad_x_x,grad_x_y = self.magnitude(x)
        # abs_grad_x = torch.sqrt(grad_x[:,:,0,:,:]**2+grad_x[:,:,1,:,:]**2)
        # print(abs_grad_x)
        grad_y,grad_y_x,grad_y_y = self.magnitude(y)
        # abs_grad_y = torch.sqrt(grad_y[:,:,0,:,:]**2+grad_y[:,:,1,:,:]**2)
#        dq = DQ(use_cuda=self.use_cuda)
#          # Mean Value Regularisation
        window_size = 11
        window = gaussian_window(window_size,1.5,y.shape[1])
        if self.use_cuda:
            window = window.cuda()
        mu_y = F.conv2d(y,window,padding = 11//2,groups = y.shape[1])
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.L1Loss(reduction='mean')
        # need padding to ensure output is the same size as shape
        loss_func = self.loss_func
        try:
            loss_func.use_cuda = use_cuda
        except:
            print("Module does not contain manual use_cuda attribute. Continuing with the assumption that input and target tensors are loaded on same device")
        if self.use_cuda:
            grad_x_x = grad_x_x.cuda()
            grad_y_x = grad_y_x.cuda()
            grad_x_y = grad_x_y.cuda()
            grad_y_y = grad_y_y.cuda()
        loss = 0.5*(loss_func(grad_x_x,grad_y_x)+ loss_func(grad_x_y,grad_y_y)) + self.lam*l1(x,mu_y)
        if self.orientation:
                loss += loss_func(self.orx(x),self.orx(y))
        return loss
