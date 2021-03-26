# MDF
MDF loss for image reconstruction algorithms

The jupyter notebook example.ipynb has a snipet of the code required to use the loss.


Abstract

Central to the application of neural networks in image restoration problems, such as single image super resolution, is the choice of a loss function that encourages natural and perceptually pleasing results. A popular choice for a loss function is a pre-trained network, such as VGG and LPIPS, which is used as a feature extractor for computing the difference between restored and reference images. However, such an approach has multiple drawbacks: it is computationally expensive, requires regularization and hyper-parameter tuning, and involves a large network trained on an unrelated task. In this work, we explore the question of what makes a good loss function for an image restoration task. First, we observe that a single natural image is sufficient to train a lightweight feature extractor that outperforms state-of-the-art loss functions in single image super resolution, denoising, and JPEG artefact removal. We propose a novel Multi-Scale Discriminative Feature (MDF) loss comprising a series of discriminators, trained to penalize errors introduced by a generator. Second, we show that an effective loss function does not have to be a good predictor of perceived image quality, but instead needs to be specialized in identifying the distortions for a given restoration method.

We provide a comprehensive comparison of qualitative results for different loss functions across different applications. To begin with, we show results for two Single Image Super-Resolution (SISR) networks, namely, Enhanced Deep Super-Resolution (EDSR) and Super-Resolution ResNet (SR-ResNet). Further, we show the results for the applications of image denoising and JPEG artefact removal.

Project Webpage:

https://www.cl.cam.ac.uk/research/rainbow/projects/mdf/


To run a simple example

```python
import torch as pt
import torch.optim as optim
import imageio
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

from mdfloss import MDFLoss


# Set parameters
cuda_available = False
epochs = 25
application = 'JPEG'
image_path = './misc/i10.png'

if application =='SISR':
    path_disc = "./weights/Ds_SISR.pth"
elif application == 'Denoising':
    path_disc = "./weights/Ds_Denoising.pth"
elif application == 'JPEG':
    path_disc = "./weights/Ds_JPEG.pth"

# Read reference images
imgr = imageio.imread(image_path)
imgr = pt.from_numpy(imageio.core.asarray(imgr/255.0))
imgr = imgr.type(dtype=pt.float64)
imgr = imgr.permute(2,0,1)
imgr = imgr.unsqueeze(0).type(pt.FloatTensor)

# Create a noisy image 
imgd = pt.rand(imgr.size())

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

# Iterate over the epochs optimizing for the noisy image
for ii in range(0,epochs):
    
    optimizer.zero_grad()
    loss = criterion(imgrb,imgdb,8,1) 
    print("Epoch: ",ii," loss: ", loss.item())
    loss.backward()
    optimizer.step()

```

If using, please cite:

```
@INPROCEEDINGS{mustafa2021,
    title={Active Sampling for Pairwise Comparisons via Approximate Message Passing and Information Gain Maximization},
    author={Aliaksei Mikhailiuk and Clifford Wilmot and Maria Perez-Ortiz and Dingcheng Yue and Rafal Mantiuk},
    booktitle={2020 IEEE International Conference on Pattern Recognition (ICPR)}, 
    year={2021},
    month={Jan},
}
```
