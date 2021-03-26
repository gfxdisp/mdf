# MDF
MDF loss for image reconstruction algorithms

The jupyter notebook example.ipynb has a snipet of the code required to use the loss.


Abstract

Central to the application of neural networks in image restoration problems, such as single image super resolution, is the choice of a loss function that encourages natural and perceptually pleasing results. A popular choice for a loss function is a pre-trained network, such as VGG and LPIPS, which is used as a feature extractor for computing the difference between restored and reference images. However, such an approach has multiple drawbacks: it is computationally expensive, requires regularization and hyper-parameter tuning, and involves a large network trained on an unrelated task. In this work, we explore the question of what makes a good loss function for an image restoration task. First, we observe that a single natural image is sufficient to train a lightweight feature extractor that outperforms state-of-the-art loss functions in single image super resolution, denoising, and JPEG artefact removal. We propose a novel Multi-Scale Discriminative Feature (MDF) loss comprising a series of discriminators, trained to penalize errors introduced by a generator. Second, we show that an effective loss function does not have to be a good predictor of perceived image quality, but instead needs to be specialized in identifying the distortions for a given restoration method.

We provide a comprehensive comparison of qualitative results for different loss functions across different applications. To begin with, we show results for two Single Image Super-Resolution (SISR) networks, namely, Enhanced Deep Super-Resolution (EDSR) and Super-Resolution ResNet (SR-ResNet). Further, we show the results for the applications of image denoising and JPEG artefact removal.

Webpage:

https://www.cl.cam.ac.uk/research/rainbow/projects/mdf/
