import torch
import torch.optim as optim
import imageio
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

from mdfloss import MDFLoss

# Set parameters
cuda_available = True
epochs = 25
application = 'JPEG'
image_path = './misc/i10.png'

if application =='SISR':
    path_disc = "./weights/Ds_SISR.pth"
elif application == 'Denoising':
    path_disc = "./weights/Ds_Denoising.pth"
elif application == 'JPEG':
    path_disc = "./weights/Ds_JPEG.pth"

#%% Read reference images
imgr = imageio.imread(image_path)
imgr = torch.from_numpy(imageio.core.asarray(imgr/255.0))
imgr = imgr.type(dtype=torch.float64)
imgr = imgr.permute(2,0,1)
imgr = imgr.unsqueeze(0).type(torch.FloatTensor)

# Create a noisy image 
imgd = torch.rand(imgr.size())

# Save the original state
imgdo = imgd.detach().clone()

if cuda_available:
    imgr = imgr.cuda()
    imgd = imgd.cuda()

# Convert images to variables to support gradients
imgrb = Variable( imgr, requires_grad = False)
imgdb = Variable( imgd, requires_grad = True)

optimizer = optim.Adam([imgdb], lr=0.1)

# Initialise the loss
criterion = MDFLoss(path_disc, cuda_available=cuda_available)

# Iterate over the epochs optimizing for the noisy image
for ii in range(0,epochs):
    
    optimizer.zero_grad()
    loss = criterion(imgrb,imgdb) 
    print("Epoch: ",ii," loss: ", loss.item())
    loss.backward()
    optimizer.step()
 
    

# Convert images to numpy
imgrnp = imgr.cpu().squeeze(0).permute(1,2,0).data.numpy()
imgdnp = imgd.cpu().squeeze(0).permute(1,2,0).data.numpy()
imgdonp = imgdo.cpu().squeeze(0).permute(1,2,0).data.numpy()


# Plot optimization results
fig, axs = plt.subplots(1, 3,figsize=(45,15))


axs[0].imshow(imgdonp)
axs[0].set_title('Noisy image',fontsize=48)
axs[1].imshow(imgdnp)
axs[1].set_title('Recovered image',fontsize=48)
axs[2].imshow(imgrnp)
axs[2].set_title('Reference image',fontsize=48)

# Remove the ticks from the axis
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
