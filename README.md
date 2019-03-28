# Wasserstein_autoencoder

This is a Pytorch implementation of the Wasserstein Autoencoder by Tolsitkhin et al. 2018 https://openreview.net/pdf?id=HkL7n1-0b.

Note:
  1. Uses a wasserstein GAN (Gulrajani et al. 2017) type discriminator on the latent space to compute the divergence term in   Wasserstein Autoencoder.
  2. The encoder and decoder are convolutional networks that down/up sample data as in DCGAN (Radford et al. 2015).
  3. # The input data are mfcc features computed from the different channels of clinical EEG data.
  
  
