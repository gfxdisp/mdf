# TODO:
Add requirements 
Update citation

# Multi-Scale Discriminative Feature Loss

This repository provides code for Multi-Scale Discriminative Feature (MDF) loss for image reconstruction algorithms.

## Description

Central to the application of neural networks in image restoration problems, such as single image super resolution, is the choice of a loss function that encourages natural and perceptually pleasing results. We provide a lightweight feature extractor that outperforms state-of-the-art loss functions in single image super resolution, denoising, and JPEG artefact removal. We propose a novel Multi-Scale Discriminative Feature (MDF) loss comprising a series of discriminators, trained to penalize errors introduced by a generator. For further information please refer to the [project webpage](https://www.cl.cam.ac.uk/research/rainbow/projects/mdf/).

## Usage

The code runs in Python3 and Pytorch.

First install the dependencies by running:

```
pip3 install -r requirements.txt
```

To run a simple example, optimizing image pixels:

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
    loss = criterion(imgrb,imgdb) 
    print("Epoch: ",ii," loss: ", loss.item())
    loss.backward()
    optimizer.step()

```


## Citing

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
## Acknowledgement

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement N◦ 725253–EyeCode).
