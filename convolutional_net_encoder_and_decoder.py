import numpy as np
import torch
from torch import nn
from itertools import chain
from torch.autograd import grad as torch_grad
import joblib
import gc
import pdb
from torch.nn.utils import spectral_norm

class conv_encoder(nn.Module):
    def __init__(self, nz = 10, nchans=6, ngf = 128):
        super(conv_encoder, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # input shape is (N, 6, 13, 425)
            nn.Conv2d(nchans, ngf*8, kernel_size=(3, 5), stride=(1,2)),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # shape is (N, ngf*4, 11, 211)
            nn.Conv2d(ngf*8, ngf*4, kernel_size=(5, 5), stride=(1,2)),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # shape is (N, ngf*2, 7, 104)
            nn.Conv2d(ngf*4, ngf*2, kernel_size=(3, 4), stride=(1,2)),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # shape is (N, ngf, 5, 51)
            nn.Conv2d(ngf*2, ngf, kernel_size=(4, 5), stride=(1,2)),
            # shape is (N, ngf, 2, 24)
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.lin1= nn.Linear(ngf*2*24, 200)
        self.lin2 = nn.Linear(200, nz)
        self.relu = nn.ReLU(True)
        self.selu = nn.SELU(True)
    def forward(self, x):
        output = self.main(x)
        output = output.view(output.size(0), -1)
        output = self.relu(self.lin1(output))
        output = self.lin2(output)
        return output
    
class conv_decoder(nn.Module):
    def __init__(self, nz = 10, nchans=6, ngf = 128):
        super(conv_decoder, self).__init__()
        self.nz = nz
        self.in_lin_map = nn.Linear(nz,50)
        self.main = nn.Sequential(
            # input shape is (N, 1, 1, 50)
            nn.ConvTranspose2d(1, ngf*4, kernel_size=(3, 4), stride=(1,2)),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # shape is (N, ngf*4, 3, 102)
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=(4, 6), stride=(1,2)),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # shape is (N, ngf*2, 6, 208)
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=(4, 4), stride=(1,2)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # shape is (N, ngf, 9, 418)
            nn.ConvTranspose2d(ngf, nchans, kernel_size=(5, 8), stride=(1,1))
            # shape is (N, 6, 13, 425)
        )   
    def forward(self, x):
        x = self.in_lin_map(x)
        x = torch.unsqueeze(x,dim=1)
        x = torch.unsqueeze(x,dim=2)
        output = self.main(x)
        return output
    
    
class ResBlock(nn.Module):
    ''' Simple dense net residual block used in latent space discriminators '''
    def __init__(self, nz):
        super(ResBlock, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(nz, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            
            nn.Linear(100, nz),
            nn.BatchNorm1d(nz),
            nn.ReLU(True)
        )
    def forward(self, z):
        res_out = self.main(z)
        return res_out + z
    
    
class latent_classifier(nn.Module):
    ''' A classifier on encoded latent variables '''
    def __init__(self, nz, nclasses):
        super(latent_classifier, self).__init__()
        self.nz = nz
        self.res1 = ResBlock(nz)
        self.res2 = ResBlock(nz)
        self.res3 = ResBlock(nz)
        self.res4 = ResBlock(nz)
        self.lin = nn.Linear(nz, nclasses)
    def forward(self, z):
        res_out = self.res3(self.res2(self.res1(z)))
        return self.lin(self.res4(res_out))
    
    
class latent_discriminator(nn.Module):
    ''' Discriminator for WGAN '''
    def __init__(self, nz):
        super(latent_discriminator, self).__init__()
        self.nz = nz
        self.res1 = ResBlock(nz)
        self.res2 = ResBlock(nz)
        self.lin = nn.Linear(nz, 1)
    def forward(self, z):
        z = self.res1(z)
        z = self.res2(z)
        return self.lin(z)
    
    