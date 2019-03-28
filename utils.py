from adverserial_autoencoder import *
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler


############ FUNCTIONS FOR TRAINING CLASSIFIER #############
def train_classifier(clf, aae_model, classifier_dataloader, optimizer, nbatches, nepochs = 5, log_every = 50):
    total_loss = []
    for epoch in range(nepochs):
        per_minibatch_loss = [None for _ in range(nbatches)]
        per_minibatch_accuracy = [None for _ in range(nbatches)]
        for i,data in enumerate(classifier_dataloader):
            x, y  = data
            yhat = clf(aae_model.enc(x))
            optimizer.zero_grad()
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            per_minibatch_loss[i] = loss.item()
            # make a prediction
            predicted_y = m(yhat)
            predicted_y = predicted_y.argmax(dim=1)
            per_minibatch_accuracy[i] = balanced_accuracy_score(y.numpy(), predicted_y.numpy())
            if i%log_every == log_every-1:
                print('epoch: %d, [%d minibatches] loss = %.2f +/- %.2f' % (epoch, i, np.mean(per_minibatch_loss[i-log_every+1:i]),
                                                                                              np.std(per_minibatch_loss[i-log_every+1:i])))
                print('epoch: %d, [%d minibatches] accuracy = %.2f +/- %.2f' % (epoch, i, np.mean(per_minibatch_accuracy[i-log_every+1:i]),
                                                                                              np.std(per_minibatch_accuracy[i-log_every+1:i])))
        total_loss.append(per_minibatch_accuracy)
    return total_loss


def test_classifier(clf, aae_model, validation_dataloader, log_every = 50):
    test_labels_and_predictions = []
    m = nn.Softmax(dim=1)
    for i,data in enumerate(validation_dataloader):
        x, y  = data
        yhat = clf(aae_model.enc(x))
        # make a prediction
        predicted_y = m(yhat)
        predicted_y = predicted_y.argmax(dim=1)
        test_labels_and_predictions.append((y.numpy(), predicted_y.numpy()))
    return test_labels_and_predictions


def load_learned_aae_and_clf(aaepath, optspath, clfpath, training_data_path):
    # load training and validation data
    data = joblib.load(training_data_path)
    # load opts dictionary
    opts = joblib.load(optspath)['opts']
    # setup validation dataset and dataloader (training is same as before)
    val_dataset = TensorDataset(torch.from_numpy(data['Xval']).float(), torch.from_numpy(data['yval']).long())
    del data
    val_dataloader = DataLoader(val_dataset, batch_size = opts['batch_size'], num_workers=0, drop_last=True)
    # load aae model
    aae = adverserial_autoencoder(val_dataloader, val_dataset.__len__(), 
                                      model_savepath = '', 
                                      downsample_to = (opts['downsample_to_H'], opts['downsample_to_W']),
                                      nz = opts['nz'], ngf = opts['ngf'], ncrit_steps = opts['ncrit_steps'],
                                      grad_penalty = opts['grad_penalty'], batch_size = opts['batch_size'],
                                      lr = opts['learning_rate'], disc_weight_decay = opts['Disc_weight_decay'],
                                     WGAN_loss_lambda = opts['WGAN_loss_lambda'])
    aae.load_state_dict(torch.load(aaepath))
    aae.eval()
    # now load clf
    latent_clf = latent_classifier(nz = opts['nz'], nclasses = len(opts['label_types']))
    latent_clf.load_state_dict(torch.load(clfpath))
    latent_clf.eval()
    return aae, latent_clf, val_dataloader


def setup_and_train_classifier(aae, aae_dataloader, X, y, nz = 16, nclasses = 3, 
                               learning_rate = 1e-3, weight_decay = 1e-3):
    aae.eval()
    m = nn.Softmax(dim=1)
    # setup validation dataset and dataloader (training is same as before)
    eeg_val_dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    eeg_val_dataloader = DataLoader(eeg_val_dataset, batch_size = args.batch_size, num_workers=2, drop_last=True)
    # setup classifier
    latent_clf = latent_classifier(nz = nz, nclasses = nclasses)
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(latent_clf.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    # train classifier
    nbatches = aae_dataset.__len__() // args.batch_size
    total_loss = train_classifier(latent_clf, aae, aae_dataloader, nbatches, args.nepochs, args.log_every)
    
    # validate classifier
    latent_clf.eval()
    labels_and_predictions = test_classifier(latent_clf, aae, eeg_val_dataloader, 50)
    true_y = [y[0] for y in labels_and_predictions]
    pred_y = [y[1] for y in labels_and_predictions]
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    val_score = balanced_accuracy_score(true_y, pred_y)
    print('\n ...... VALIDATION SCORE = %f ...... '%(val_score))
    return latent_clf, val_score

    
def plot_and_save_loss(loss, savepath, savestr):
    plt.figure(figsize=(10,8))
    plt.plot(loss, '-.k')
    plt.xlabel('minibatches')
    plt.ylabel('loss')
    plt.savefig(os.path.join(savepath, savestr + '.png'), format = 'png', dpi = 100)
    
def load_and_encode(aaepath, training_data_path, optspath):
    '''
    Encode the MFCC data into the latent space and return the infered latent variables and the labels
    '''
    # load training and validation data
    data = joblib.load(training_data_path)
    # load opts dictionary
    opts = joblib.load(optspath)['opts']
    # setup validation dataset and dataloader (training is same as before)
    train_dataset = TensorDataset(torch.from_numpy(data['Xtrain']).float(), torch.from_numpy(data['ytrain']).long())
    yout = data['ytrain']
    del data
    train_dataloader = DataLoader(train_dataset, batch_size = opts['batch_size'], num_workers=0, drop_last=True)
    # load aae model
    aae = adverserial_autoencoder(train_dataloader, len(train_dataloader), 
                                      model_savepath = '', 
                                      downsample_to = (opts['downsample_to_H'], opts['downsample_to_W']),
                                      nz = opts['nz'], ngf = opts['ngf'], ncrit_steps = opts['ncrit_steps'],
                                      grad_penalty = opts['grad_penalty'], batch_size = opts['batch_size'],
                                      lr = opts['learning_rate'], disc_weight_decay = opts['Disc_weight_decay'],
                                     WGAN_loss_lambda = opts['WGAN_loss_lambda'])
    aae.load_state_dict(torch.load(aaepath))
    aae.eval()
    
    Zout = np.zeros((len(train_dataloader), opts['batch_size'], opts['nz']))
    for i,(x,y) in enumerate(train_dataloader):
        z = aae.enc(x)
        Zout[i] = z.detach().numpy()
    # collapse first two dimensions
    Zout = np.reshape(Zout, (len(train_dataloader)*opts['batch_size'], opts['nz']))
    return Zout, yout


def encode(dataloader, aaemodel, opts):
    Zout = np.zeros((len(dataloader), opts['batch_size'], opts['nz']))
    for i,(x,y) in enumerate(dataloader):
        z = aaemodel.enc(x)
        Zout[i] = z.detach().numpy()
    return Zout