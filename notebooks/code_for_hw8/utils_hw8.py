import numpy as np
import itertools
import math as m
from matplotlib import pyplot as plt

import torch
from torch.nn import Linear, ReLU, Softmax, Sequential, CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

######################################################################
# Plotting Functions
######################################################################

def plot_heat(X, y, model, res = 200):
    eps = .1
    X, y = X.numpy(), y.numpy()
    xmin = np.min(X[:,0]) - eps; xmax = np.max(X[:,0]) + eps
    ymin = np.min(X[:,1]) - eps; ymax = np.max(X[:,1]) + eps
    ax = tidyPlot(xmin, xmax, ymin, ymax, xlabel = 'x', ylabel = 'y')
    xl = np.linspace(xmin, xmax, res)
    yl = np.linspace(ymin, ymax, res)
    xx, yy = np.meshgrid(xl, yl, sparse=False)
    data = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    zz = np.argmax(model(data).detach().numpy(), axis=1)
    im = ax.imshow(np.flipud(zz.reshape((res,res))), interpolation = 'none',
                   extent = [xmin, xmax, ymin, ymax],
                   cmap = 'viridis')
    plt.colorbar(im)
    for yi in set([int(_y) for _y in set(y)]):
        color = ['r', 'g', 'b'][yi]
        marker = ['X', 'o', 'v'][yi]
        cl = np.where(y==yi)
        ax.scatter(X[cl,0], X[cl,1], c = color, marker = marker, s=80,
                   edgecolors = 'none')
    return ax

def tidyPlot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_separator(ax, th, th_0):
    xmin, xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    pts = []
    eps = 1.0e-6
    # xmin boundary crossing is when xmin th[0] + y th[1] + th_0 = 0
    # that is, y = (-th_0 - xmin th[0]) / th[1]
    if abs(th[1,0]) > eps:
        pts += [np.array([x, (-th_0 - x * th[0,0]) / th[1,0]]) \
                                                        for x in (xmin, xmax)]
    if abs(th[0,0]) > 1.0e-6:
        pts += [np.array([(-th_0 - y * th[1,0]) / th[0,0], y]) \
                                                         for y in (ymin, ymax)]
    in_pts = []
    for p in pts:
        if (xmin-eps) <= p[0] <= (xmax+eps) and \
           (ymin-eps) <= p[1] <= (ymax+eps):
            duplicate = False
            for p1 in in_pts:
                if np.max(np.abs(p - p1)) < 1.0e-6:
                    duplicate = True
            if not duplicate:
                in_pts.append(p)
    if in_pts and len(in_pts) >= 2:
        # Plot separator
        vpts = np.vstack(in_pts)
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Plot normal
        vmid = 0.5*(in_pts[0] + in_pts[1])
        scale = np.sum(th*th)**0.5
        diff = in_pts[0] - in_pts[1]
        dist = max(xmax-xmin, ymax-ymin)
        vnrm = vmid + (dist/10)*(th.T[0]/scale)
        vpts = np.vstack([vmid, vnrm])
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Try to keep limits from moving around
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    else:
        print('Separator not in plot range')

def plot_decision(data, cl, diff=False):
    layers = archs(cl)[0]
    X, y, model = run_pytorch_2d(data, layers, 10, trials=1, verbose=False, display=False)
    ax = plot_heat(X,y,model)
    W = layers[0].get_weights()[0]
    W0 = layers[0].get_weights()[1].reshape((cl,1))
    if diff:
        for i,j in list(itertools.combinations(range(cl),2)):
            plot_separator(ax, W[:,i:i+1] - W[:,j:j+1], W0[i:i+1,:] - W0[j:j+1,:])
    else:
        for i in range(cl):
            plot_separator(ax, W[:,i:i+1], W0[i:i+1,:])
    plt.show()

def make_iter(X, y, batch_size=1):
    X, y = torch.FloatTensor(X), torch.FloatTensor(y)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)

def set_weights(module, weights):
    weights = torch.FloatTensor(weights.reshape(module.weight.shape))
    module.weight = torch.nn.Parameter(weights)

def set_bias(module, bias):
    module.bias = torch.nn.Parameter(torch.FloatTensor([bias]))


def call_model(mode, model, data_iter, optimizer, criterion, model_reg=None):
    epoch_loss = []
    hits = []
    items = []
    
    # This is to manually regularize model, should return a value that is 
    # a function of the model parameters.
    if model_reg == None:
        model_reg = lambda x: 0

    if mode == 'train':
        model.train()
        grad_mode = torch.enable_grad()
    else:
        model.eval()
        grad_mode = torch.no_grad()

    with grad_mode:

        for batch in data_iter:
            X, y = batch

            if mode == 'train':
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward
            y_hat = model(X)
            if type(criterion) == CrossEntropyLoss:
                batch_loss = criterion(y_hat, y.long()) + model_reg(model)
            else:
                batch_loss = criterion(y_hat, y.float()) + model_reg(model)

            if mode == 'train':
                # backward + optimize
                batch_loss.backward()
                optimizer.step()

            epoch_loss.append(batch_loss.item())
            if type(criterion) == CrossEntropyLoss:
                hits.append((y_hat.argmax(1) == y.long()).sum())
            else:
                hits.append((((y_hat - y.float()) ** 2.).sum().item()))
            items.append(X.shape[0])

        loss = np.sum(epoch_loss)/np.sum(items)
        acc_score = np.sum(hits)/np.sum(items)
        return loss, acc_score

def model_fit(model, train_iter, epochs, optimizer, criterion,
              validation_iter, history, verbose, model_reg=None):

  av_train_loss, av_train_acc, av_vali_loss, av_vali_acc = [], [], [], []
  for epoch in range(epochs):
      train_loss, train_acc_score = call_model('train', model, train_iter, optimizer, criterion, model_reg=model_reg)
      vali_loss, vali_acc_score  = call_model('vali', model, validation_iter, optimizer, criterion)
      if verbose: print("epoch: {} | TRAIN: loss {} acc {} | VALI: loss {} acc {}".format(epoch,
                                                                 round(train_loss, 5), 
                                                                 round(train_acc_score, 5), 
                                                                 round(vali_loss, 5), 
                                                                 round(vali_acc_score, 5)))
      if history is not None:
          history['epoch_loss'].append(train_loss)
          history['epoch_val_loss'].append(vali_loss)
          history['epoch_acc'].append(train_acc_score)
          history['epoch_val_acc'].append(vali_acc_score)
      av_train_loss.append(train_loss)
      av_train_acc.append(train_acc_score)
      av_vali_loss.append(vali_loss)
      av_vali_acc.append(vali_acc_score)
    
  return ((np.mean(av_train_loss), np.mean(av_train_acc)), 
         (np.mean(av_vali_loss), np.mean(av_vali_acc)))

def model_evaluate(model, test_iter, criterion):
    vali_loss, vali_acc_score = call_model('vali', model, test_iter, None, criterion)
    return vali_loss, vali_acc_score


def run_pytorch(train_iter, val_iter, test_iter, layers, epochs,
                verbose=True, history=None, loss='xent'):
    # Model specification
    model = Sequential(*layers)

    # Define the optimization
    optimizer = Adam(model.parameters())
    if loss == 'xent':
        criterion = CrossEntropyLoss()
    else:
        criterion = MSELoss()
    
    # Fit the model
    train_m, vali_m = model_fit(model, train_iter, epochs=epochs, 
                                optimizer=optimizer, criterion=criterion,
                                validation_iter=val_iter,
                                history=history, verbose=verbose)
    if verbose: print()
    
    (train_loss, train_acc) = train_m
    (vali_loss, val_acc) = vali_m
    
    # Evaluate the model on test data, if any
    if test_iter is not None:
        test_loss, test_acc = model_evaluate(model, test_iter, criterion)
        print ("\nLoss on test set:"  + str(test_loss) + " Accuracy on test set: " + str(test_acc))
    else:
        test_acc = None
    return model, val_acc, test_acc



# The name is a string such as "1" or "Xor"
def run_pytorch_2d(data_name, layers, epochs, split=0.25, display=True,
                   verbose=True, trials=1, batch_size=32):
    print('Pytorch FC: dataset=', data_name)
    (train_dataset_path, val_dataset_path, test_dataset_path) = dataset_paths(data_name)
    # Load the datasets
    train_iter, num_classes = get_data_loader(train_dataset_path, batch_size)
    val_iter, num_classes = get_data_loader(val_dataset_path, batch_size)
    test_iter, num_classes = get_data_loader(test_dataset_path, batch_size)
    
    if val_iter is None:
        # Use split
        print("Use split", train_iter)
        assert split > 0, '`split` must be > 0'
        train_iter, val_iter,  num_classes = get_data_loader(train_dataset_path, batch_size, split)

    val_acc, test_acc = 0, 0
    X_train = torch.cat([batch.X for batch in train_iter], 0)
    y_train = torch.cat([batch.y for batch in train_iter], 0)
    
    for trial in range(trials):
        trial_history = {'epoch_loss': [], 'epoch_val_loss': [],
               'epoch_acc': [], 'epoch_val_acc': []}
    
        if verbose: print("\n")
        print(f'# Trial {trial}')
        
        # Run the model
        model, vacc, tacc, = run_pytorch(train_iter, val_iter, test_iter, 
                                         layers, epochs, split=split,
                                         verbose=verbose, history=trial_history)

        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
        if display:
            # plot classifier landscape on training data
            plot_heat(X_train, y_train, model)
            plt.title('Training data')
            plt.show()
            if test_iter is not None:
                # plot classifier landscape on testing data
                X_test = torch.cat([batch.X for batch in test_iter], 0)
                y_test = torch.cat([batch.y for batch in test_iter], 0)
                plot_heat(X_test, y_test, model)
                plt.title('Testing data')
                plt.show()
            # Plot epoch loss
            plt.figure(facecolor="white")
            plt.plot(range(epochs), trial_history['epoch_loss'], label='epoch_train_loss')
            plt.plot(range(epochs), trial_history['epoch_val_loss'], label='epoch_val_loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('Epoch val_loss and loss')
            plt.legend()
            plt.show()
            # Plot epoch accuracy
            plt.figure(facecolor="white")
            plt.plot(range(epochs), trial_history['epoch_acc'], label='epoch_train_acc')
            plt.plot(range(epochs), trial_history['epoch_val_acc'], label='epoch_val_acc')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend()
            plt.title('Epoch val_acc and acc')
            plt.show()
    if val_acc:
        print ("\nAvg. validation accuracy:"  + str(val_acc/trials))
    if test_acc:
        print ("\nAvg. test accuracy:"  + str(test_acc/trials))
        
   
    return X_train, y_train, model