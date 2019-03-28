from models import *
import pdb
import matplotlib.pyplot as plt
    
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type = str)
parser.add_argument('--split_training_datapath', type = str, default = '')
parser.add_argument('--model_savepath', type = str, default = '')
parser.add_argument('--wae_loadpath', type = str, default = '')
parser.add_argument('--clf_loadpath', type = str, default = '')
parser.add_argument('--weighted_sampling', action = 'store_true')
parser.add_argument('--clf_loss_weighting', action = 'store_true')
parser.add_argument('--normalize', action = 'store_true')
parser.add_argument('--train_val_split_ratio', default = 0.8)
parser.add_argument('--nz', type=int, default = 16)
parser.add_argument('--downsample_to_H', type = int, default = 5)
parser.add_argument('--downsample_to_W', type = int, default = 20)
parser.add_argument('--WGAN_loss_lambda', type = float, default = 10.)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--learning_rate', type = float, default = 1e-4)
parser.add_argument('--grad_penalty', type = float, default = 10)
parser.add_argument('--ncrit_steps', type = int, default = 5)
parser.add_argument('--Disc_weight_decay', type = float, default = 1e-3)
parser.add_argument('--ngf', type = int, default = 30)
parser.add_argument('--nepochs', type = int, default = 10)
parser.add_argument('--log_every', type = int, default = 100)

opts = {'weighted_sampling': True,
       'clf_loss_weighting': True,
        'normalize': False,
        'normalizer': [],
       'train_val_split_ratio': 0.8,
       'nz': 16,
        'ngf': 40,
       'downsample_to_H': 5,
       'downsample_to_W': 20,
       'batch_size': 32,
        'WGAN_loss_lambda':10., 
        'seed': 1,
       'learning_rate': 1e-4,
       'grad_penalty': 10.,
       'ncrit_steps': 7,
       'Disc_weight_decay': 1e-3,
       'nepochs': 10,
       'log_every': 100,
       'label_types': np.array([0, 1, 2]),
       'loss_weights': np.zeros(3),
       'resample_weights': [],
       'Ntrain_samples': 0,
       'Nval_samples' : 0,
       'Ntrain_batches': 0,
       'Nval_batches': 0}



if __name__ == '__main__':
    args = parser.parse_args()
    opts['weighted_sampling'] = args.weighted_sampling
    opts['clf_loss_weighting'] = args.clf_loss_weighting
    opts['normalize'] = args.normalize
    opts['train_val_split_ratio'] = args.train_val_split_ratio
    opts['nz'] = args.nz
    opts['ngf'] = args.ngf
    opts['downsample_to_H'] = args.downsample_to_H
    opts['downsample_to_W'] = args.downsample_to_W
    opts['WGAN_loss_lambda'] = args.WGAN_loss_lambda
    opts['batch_size'] = args.batch_size
    opts['learning_rate'] = args.learning_rate
    opts['grad_penalty'] = args.grad_penalty
    opts['ncrit_steps'] = args.ncrit_steps
    opts['Disc_weight_decay'] = args.Disc_weight_decay
    opts['nepochs'] = args.nepochs
    opts['log_every'] = args.log_every
    
    # set seed 
    torch.manual_seed(opts['seed'])
    
    # change model path to include todays date
    datstr = str(datetime.now()).replace(':', '-')
    args.model_savepath = args.model_savepath + '/' + datstr
    if not os.path.exists(args.model_savepath):
        os.makedirs(args.model_savepath, exist_ok=True)
    
    if len(args.wae_loadpath)==0:
        # load data
        data_dict = joblib.load(args.datapath)
        labels = data_dict['labels']
        data = data_dict['data']
        # normalize ? 
        if opts['normalize']:
            data = data/np.median(data, axis=0)
            opts['normalizer'] = np.median(data, axis=0)
        del data_dict
        gc.collect()
        # make training and validation sets
        Xtrain, Xval, ytrain, yval = train_test_split(data, labels, test_size = args.train_val_split_ratio)
        del data, labels
        gc.collect()
        
        # relabel ytrain and yval (specific to current applications)
        ytrain[ytrain==4] = 2
        yval[yval==4] = 2

        # setup dataset and dataloader
        
        # setup Weighted random sampler (to upsample minority class and downsample majority class)
        nlabels, class_sample_counts = np.unique(ytrain, return_counts=True)
        opts['label_types'] = nlabels
        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        samples_weights = weights[torch.from_numpy(ytrain).long()]
        opts['resample_weights'] = samples_weights
        
        if args.weighted_sampling:
            # define new dataloader with undersampling majority class
            Rsampler = WeightedRandomSampler(
                weights=samples_weights,
                num_samples=len(samples_weights),
                replacement=True)
            train_dataset = TensorDataset(torch.from_numpy(Xtrain).float(), torch.from_numpy(ytrain).long())
            train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=2, sampler = Rsampler, drop_last=True)
        else:
            train_dataset = TensorDataset(torch.from_numpy(Xtrain).float(), torch.from_numpy(ytrain).long())
            train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=2, shuffle = True, drop_last=True)
            
        # setup validation dataset and dataloader (training is same as before)
        val_dataset = TensorDataset(torch.from_numpy(Xval).float(), torch.from_numpy(yval).long())
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers=2, drop_last=True)
        
        # save opts
        joblib.dump({'opts': opts}, args.model_savepath + '/opts.pkl')
        
        # setup WAE model
        wae = wasserstein_autoencoder(train_dataloader, train_dataset.__len__(), 
                                      model_savepath=args.model_savepath, 
                                      downsample_to = (args.downsample_to_H, args.downsample_to_W),
                                      nz = args.nz, ngf = args.ngf, ncrit_steps = args.ncrit_steps,
                                      grad_penalty = args.grad_penalty, batch_size = args.batch_size,
                                      lr = args.learning_rate, disc_weight_decay = args.Disc_weight_decay,
                                     WGAN_loss_lambda = args.WGAN_loss_lambda)
        # train the autoencoder
        recon_loss, disc_loss, gen_loss = wae.fit(nepochs = args.nepochs, log_every = args.log_every)
        
        recon_loss = np.concatenate(recon_loss)
        recon_loss = [r for r in recon_loss if r is not None]
        disc_loss = np.concatenate(disc_loss)
        disc_loss = [d for d in disc_loss if d is not None]
        gen_loss = np.concatenate(gen_loss)
        gen_loss = [g for g in gen_loss if g is not None]
        
        # save model and losses
        torch.save(wae.state_dict(), os.path.join(args.model_savepath, 'wae_full_model.pth'))
        joblib.dump({'recon_loss': recon_loss, 'disc_loss':disc_loss, 'gen_loss':gen_loss}, 
                    os.path.join(args.model_savepath, 'losses.pkl'))
        
        # make a graph for the losses
        plot_and_save_loss(recon_loss, args.model_savepath, 'recon_loss')
        plot_and_save_loss(disc_loss, args.model_savepath, 'disc_loss')
        plot_and_save_loss(gen_loss, args.model_savepath, 'gen_loss')
        
        
    wae.eval()
    m = nn.Softmax(dim=1)
    
    print('\n ####### A TRAINING CLASSIFIER ON THE LATENT SPACE ######## \n')
    if len(args.clf_loadpath)==0:
        # setup classifier
        latent_clf = latent_classifier(nz = args.nz, nclasses = len(nlabels))
        # loss function
        if args.clf_loss_weighting:
            loss_weights = weights/weights.sum()
            opts['loss_weights'] = loss_weights
            criterion = nn.CrossEntropyLoss(weight = loss_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(latent_clf.parameters(), lr = args.learning_rate, weight_decay = args.Disc_weight_decay)
        
        # train classifier
        nbatches = train_dataset.__len__() // args.batch_size
        opts['Ntrain_batches']  = nbatches
        total_loss = train_classifier(latent_clf, wae, train_dataloader, optimizer, nbatches, args.nepochs, args.log_every)
         # save classifier
        torch.save(latent_clf.state_dict(), os.path.join(args.model_savepath, 'clf_full_model.pth'))
    else:
        latent_clf = latent_classifier(nz = args.nz, nclasses = nlabels)
        latent_clf.load_state_dict(torch.load(args.clf_loadpath))
    
    # validate classifier
    latent_clf.eval()
    opts['Nval_batches'] = val_dataset.__len__() // args.batch_size
    labels_and_predictions = test_classifier(latent_clf, wae, val_dataloader, args.log_every)
    true_y = [y[0] for y in labels_and_predictions]
    pred_y = [y[1] for y in labels_and_predictions]
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    val_score = balanced_accuracy_score(true_y, pred_y)
    
   
    joblib.dump({'val_score': val_score}, os.path.join(args.model_savepath, 'losses.pkl'))
    # save args 
    joblib.dump({'opts': opts}, args.model_savepath + '/opts.pkl')
    print('\n ...... VALIDATION SCORE = %f ...... '%(val_score))