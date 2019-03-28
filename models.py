from convolutional_net_encoder_and_decoder import *
import argparse
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime
import pdb
import os

class wasserstein_autoencoder(nn.Module):
    def __init__(self, dataloader, Nsamples, model_savepath, nz=16, downsample_to = (5,20), nchans=6, batch_size=32, ngf=128, d_noise = 0.1, 
                 lr = 1e-4, z_var = 1., WGAN_loss_lambda = 10., grad_penalty = 10., ncrit_steps = 1, disc_weight_decay = 1e-3):
        super(adverserial_autoencoder, self).__init__()
        # models
        self.enc = conv_encoder(nz, nchans, ngf)
        self.dec = conv_decoder(nz, nchans, ngf)
        self.latent_disc = latent_discriminator(nz)
        self.dataloader = dataloader
        self.recon_loss = nn.MSELoss(reduction='elementwise_mean')
        self.optim_enc = torch.optim.Adam(self.enc.parameters(),lr=lr)
        self.optim_dec = torch.optim.Adam(chain(self.dec.parameters(), self.enc.parameters()), lr=lr)
        self.optim_disc = torch.optim.Adam(self.latent_disc.parameters(),lr=lr, weight_decay = disc_weight_decay)
        self.BCEcrit = nn.BCEWithLogitsLoss(reduction='elementwise_mean')
        self.adapavgpool = nn.AdaptiveAvgPool2d(downsample_to)
        # hyper-parameters
        self.nz = nz # latent space dimensionality
        self.Nsamps = Nsamples
        self.nbatches = Nsamples // batch_size
        self.batch_size = batch_size
        self.d_noise = d_noise
        self.gp_weight = grad_penalty
        self.WGAN_loss_lambda = WGAN_loss_lambda
        self.z_var = z_var
        # number of iterations to train discriminator before training encoder
        self.ncrit_steps = ncrit_steps
        self.model_savepath = model_savepath
        # L2 weight decay penalty for discriminator
        self.weight_decay = disc_weight_decay
        
    def downsampled_MSE(self, xhat, x):
        # downsample both to required size
        x = self.adapavgpool(x)
        xhat = self.adapavgpool(xhat)
        # find mse 
        return self.recon_loss(xhat, x)
    
    def sample_z(self, z_var=[], required_size = []):
        if len(required_size)==0:
            required_size=(self.batch_size,self.nz)
        if len(z_var)==0:
            z_var = self.z_var
        z =  torch.FloatTensor(required_size[0],required_size[1]).normal_(0, z_var)
        return z
    
    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat
    
    def infer_class_score(self, z):
        return self.latent_disc(z)
    
    def train_discriminator_WGAN_penalty(self, zsampled, zencoded):
        '''
        Guljarani et al. Improving Wasserstein GAN (2017).
        This GAN is used to compute the divergence term between the Prior P(z)
        over the latent space and the aggregated posterior Q(z) = \sum{x} Q(z|x)
        '''
        # interpolation coefficient
        eps = torch.FloatTensor(zsampled.size(0),1).uniform_(0,1)
        # get interpolated latent points
        ztilde = (eps * zsampled) + ((1 - eps) * zencoded)
        # compute discriminator scores
        score_real = self.infer_class_score(zsampled)
        score_fake = self.infer_class_score(zencoded)
        score_ztilde = self.infer_class_score(ztilde)
        # compute gradient of critic w.r.t input
        grads = torch_grad(outputs=score_ztilde, inputs=ztilde,
                           grad_outputs=torch.ones(score_ztilde.size()),
                           create_graph=True, retain_graph=True, only_inputs = True)[0]
        grads = grads.view(self.batch_size, -1)
        grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-11)
        dloss = score_fake - score_real 
        # discriminator (critic) loss
        dloss = dloss.mean() + self.gp_weight * ((grads_norm - 1) ** 2).mean()
        return dloss
            
    def gantrain_encoder(self, zencoded):
        # Now, encoder has to be trained to fool 
        # the discriminator meaning min Loss generator 
        # = -D(zencoded)
        score_fake = self.infer_class_score(zencoded)
        return -score_fake.mean()
    
    def fit(self, nepochs = 10, log_every = 1000):
        self.nepochs = nepochs
        per_epoch_reconloss = [None for _ in range(nepochs)]
        per_epoch_discloss = [None for _ in range(nepochs)]
        per_epoch_genloss = [None for _ in range(nepochs)]
        for epoch in range(nepochs):
            per_minibatch_reconloss = [None for _ in range(self.nbatches)]
            per_minibatch_discloss = [None for _ in range(self.nbatches)]
            per_minibatch_genloss = [None for _ in range(self.nbatches)]
            disc_steps = 0
            for (i, data) in enumerate(self.dataloader):
                self.optim_dec.zero_grad()
                x, y = data
                # encode and decode
                z = self.enc(x)
                xhat = self.dec(z)
                # compute reconstruction loss
                recloss = self.downsampled_MSE(xhat, x)
                # update encoder and decoder
                recloss.backward()
                self.optim_dec.step()
                per_minibatch_reconloss[i] = recloss.item()
                
                # now for adverserial loss
                # sample a bunch of z-vectors
                zsampled = self.sample_z()
                z = self.enc(x)
                # compute loss and make grad step WGAN + GP
                running_loss = 0.
                if disc_steps < self.ncrit_steps:
                    self.optim_disc.zero_grad()
                    dloss = self.WGAN_loss_lambda*self.train_discriminator_WGAN_penalty(zsampled, z)
                    dloss.backward()
                    self.optim_disc.step()
                    running_loss += dloss.item()
                    disc_steps += 1
                    per_minibatch_discloss[i] = running_loss
                    per_minibatch_genloss[i] = np.nan
                else:
                    # train encoder to fool discriminator
                    self.optim_enc.zero_grad()
                    gloss = self.WGAN_loss_lambda*self.gantrain_encoder(z)
                    gloss.backward()
                    self.optim_enc.step()
                    per_minibatch_genloss[i] = gloss.item()
                    per_minibatch_discloss[i] = np.nan
                    disc_steps = 0
                    
                if i%log_every == log_every-1:
                    # print the mean and standard deviation of loss
                    print('epoch: %d, [%d minibatches] reconstruction loss = %.2f +/- %.2f' % (epoch, i, 
                                                                                               np.mean(per_minibatch_reconloss[i-log_every+1:i]),
                                                                                              np.std(per_minibatch_reconloss[i-log_every+1:i])))
                    print('epoch: %d, [%d minibatches] disc loss = %.2f +/- %.2f' % (epoch, i, 
                                                                                               np.nanmean(per_minibatch_discloss[i-log_every+1:i]),
                                                                                              np.nanstd(per_minibatch_discloss[i-log_every+1:i])))
                    print('epoch: %d, [%d minibatches] gen loss = %.2f +/- %.2f' % (epoch, i, 
                                                                                               np.nanmean(per_minibatch_genloss[i-log_every+1:i]),
                                                                                              np.nanstd(per_minibatch_genloss[i-log_every+1:i])))
                
            per_epoch_reconloss[epoch] = per_minibatch_reconloss
            per_epoch_discloss[epoch] = per_minibatch_discloss
            per_epoch_genloss[epoch] = per_minibatch_genloss
            
            # checkpoint model
            if epoch==0:
                # check if directory exists
                if not os.path.exists(self.model_savepath):
                    os.makedirs(self.model_savepath)
            torch.save(self.state_dict(), os.path.join(self.model_savepath, 'full_model_epoch_%d.pth'%(epoch)))
        return per_epoch_reconloss, per_epoch_discloss, per_epoch_genloss
            



def plot_and_save_loss(loss, savepath, savestr):
    plt.figure(figsize=(10,8))
    plt.plot(loss, '-.k')
    plt.xlabel('minibatches')
    plt.ylabel('loss')
    plt.savefig(os.path.join(savepath, savestr + '.png'), format = 'png', dpi = 100)

    
    
    