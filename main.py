from data import loaddata
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from transformers import BertTokenizer, BertModel
import numpy as np
import sys
import re
from model import MILNCELoss, MODEL



def loss_fn(logp, target, length):
    target = target[:, :torch.max(length).data.item()].contiguous().view(-1)
    logp = logp[:, :torch.max(length).data.item()].contiguous()
    logp = logp.view(logp.size(0)*logp.size(1), logp.size(2))
    NLL_loss = NLL(logp, target)
    return NLL_loss


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def test(X_test, Xb_test, Xa_test, Xn_test, X_input_test, X_len_test, y_test, name='test'):
    model.eval()
    num_batch = int((len(y_test) + BATCH_SIZE)/BATCH_SIZE)
    total_loss = 0 
    Ybar = []
    for i in range(num_batch):
        sys.stdout.write('\r{0}/{1}'.format(i, num_batch))
        st = i * BATCH_SIZE
        ed = min((i+1)*BATCH_SIZE, len(y_test))

        x = torch.Tensor(X_test[st:ed]).to(device)
        xa = torch.from_numpy(Xa_test[st:ed]).to(device)
        xb = torch.from_numpy(Xb_test[st:ed]).to(device)
        xn = torch.from_numpy(Xn_test[st:ed]).to(device)
        x_input = torch.from_numpy(X_input_test[st:ed]).to(device)
        x_len = torch.from_numpy(X_len_test[st:ed]).to(device)
        y = torch.Tensor(y_test[st:ed]).to(device)

        src_mask = generate_square_subsequent_mask(args.textLen).to(device)
        ybar, logp, ebd1, ebd2 = model(x, xb, xn, x_input, x_len, xa, src_mask)
        aeloss = loss_fn(logp, x_input, x_len)
        ybar = torch.squeeze(ybar, 1)
        nceloss = nce_criterion(ebd1, ebd2)
        loss = nceloss + alpha * aeloss
        ybar = ybar.to('cpu')
        Ybar.extend(ybar.data.numpy())
        total_loss += loss.item()
    Ybar = np.array(Ybar)
    assert len(y_test) == len(Ybar)
    rmse = np.sqrt(mean_squared_error(y_test*(max_y-min_y) + min_y, Ybar*(max_y-min_y) + min_y))
    mae = mean_absolute_error(y_test*(max_y-min_y) + min_y, Ybar*(max_y-min_y) + min_y)
    print('{0}: total loss: {1:.4f}, RMSE {2:.4f}, MAE {3:.4f}'.format(name, total_loss, rmse, mae))

    return total_loss, rmse, y_test*(max_y-min_y)+min_y, Ybar*(max_y-min_y)+min_y
    
def train(X_train, Xb_train, Xa_train, Xn_train, X_input_train, X_len_train, y_train, name='train'):
    model.train()
    num_batch = int((len(y_train) + BATCH_SIZE)/BATCH_SIZE)
    total_loss = 0 
    Ybar = []
    for i in range(num_batch):
        sys.stdout.write('\r{0}/{1}'.format(i, num_batch))
        st = i * BATCH_SIZE
        ed = min((i+1)*BATCH_SIZE, len(y_train))
        x = torch.Tensor(X_train[st:ed]).to(device)
        xa = torch.from_numpy(Xa_train[st:ed]).to(device)
        xb = torch.from_numpy(Xb_train[st:ed]).to(device)
        xn = torch.from_numpy(Xn_train[st:ed]).to(device)
        x_input = torch.from_numpy(X_input_train[st:ed]).to(device)
        x_len = torch.from_numpy(X_len_train[st:ed]).to(device)
        y = torch.Tensor(y_train[st:ed]).to(device)
        model.zero_grad()
        src_mask = generate_square_subsequent_mask(args.textLen).to(device)
        ybar, logp, ebd1, ebd2 = model(x, xb, xn, x_input, x_len, xa, src_mask)

        aeloss = loss_fn(logp, x_input, x_len)
        nceloss = nce_criterion(ebd1, ebd2)
        ybar = torch.squeeze(ybar, 1)
        loss = nceloss + alpha * aeloss
        ybar = ybar.to('cpu')
        Ybar.extend(ybar.data.numpy())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0 and False:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, ed, len(y_train),
                100.0 * ed / len(y_train), loss.item()))

    Ybar = np.array(Ybar)
    rmse = np.sqrt(mean_squared_error(y_train*(max_y-min_y)+min_y, Ybar*(max_y-min_y)+min_y))
    mae = mean_absolute_error(y_train*(max_y-min_y)+min_y, Ybar*(max_y-min_y)+min_y)
    print('====> Epoch: {0} total loss: {1:.4f}, RMSE: {2:.4f}, MAE {3:.4f}'.format(epoch, total_loss, rmse, mae))

def test_emb(X, Xb, Xa, Xn, X_input, X_len, y, name='emb'):
    model.eval()
    num_batch = int((len(y) + BATCH_SIZE)/BATCH_SIZE)
    total_loss = 0 
    Ybar = []
    EBD1, EBD2 = [], []
    for i in range(num_batch):
        sys.stdout.write('\r{0}/{1}'.format(i, num_batch))
        st = i * BATCH_SIZE
        ed = min((i+1)*BATCH_SIZE, len(y))

        x = torch.Tensor(X[st:ed]).to(device)
        xa = torch.from_numpy(Xa[st:ed]).to(device)
        xb = torch.from_numpy(Xb[st:ed]).to(device)
        xn = torch.from_numpy(Xn[st:ed]).to(device)
        x_input = torch.from_numpy(X_input[st:ed]).to(device)
        x_len = torch.from_numpy(X_len[st:ed]).to(device)
     

        src_mask = generate_square_subsequent_mask(args.textLen).to(device)
        ybar, logp, ebd1, ebd2 = model(x, xb, xn, x_input, x_len, xa, src_mask)
        aeloss = loss_fn(logp, x_input, x_len)
        nceloss = nce_criterion(ebd1, ebd2)
        ybar = torch.squeeze(ybar, 1)
        loss = nceloss
        Ybar.extend(ybar.to('cpu').data.numpy())
        EBD1.extend(ebd1.to('cpu').data.numpy())
        EBD2.extend(ebd2.to('cpu').data.numpy())
        total_loss += loss.item()
        
    Ybar = np.array(Ybar)
    EBD1 = np.array(EBD1)
    EBD2 = np.array(EBD2)

    assert len(y) == len(Ybar)
    print('{0}: total loss: {1:.4f}'.format(name, total_loss))

    return EBD1, EBD2

def compute_metrics(x):
    sx = np.sort(-x, axis=1) #[len, len]
    d = np.diag(-x) #[len,]
    d = d[:, np.newaxis] #[len, 1]
    ind = sx - d 
    ind = np.where(ind == 0) 
    ind = ind[1] # []
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

def SVM(EMB, y, EMB_test=None, y_test=None, des=''):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.svm import LinearSVR 
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
 
    max_y, min_y = 300.0, 30.0
    if EMB_test is not None:
        X_train, X_test, y_train, y_test = EMB, EMB_test, y, y_test
    else:
        cut = int(len(y)* 0.7)
        X_train, X_test, y_train, y_test = EMB[:cut], EMB[cut:], y[:cut], y[cut:] 
    clf = SVR()
    clf.fit(X_train, y_train)
    ybar = clf.predict(X_test)
    
    mae = mean_absolute_error(ybar*(max_y - min_y) + min_y, y_test*(max_y - min_y) + min_y)
    rmse = mean_squared_error(ybar*(max_y - min_y) + min_y, y_test*(max_y - min_y) + min_y, squared=False)
    mape = mean_absolute_percentage_error(ybar*(max_y - min_y) + min_y, y_test*(max_y - min_y) + min_y)
    print('[SVM] {0}: MAE {1}, RMSE {2}, MAPE {3}.'.format(des, mae, rmse, mape))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['train', 'infer', 'baseline'],\
        default='train',help='Run mode')
    arg_parser.add_argument('--device', choices=['device', 'cpu'],\
        default='cuda',help='Device')
    arg_parser.add_argument('--epoch', default='100', type=int)
    arg_parser.add_argument('--batch_size', default='1024', type=int)
    arg_parser.add_argument('--path', default='model0.h5', type=str)
    arg_parser.add_argument('--layer', default=3, type=int)
    arg_parser.add_argument('--textLen', default=100, type=int)
    arg_parser.add_argument('--loadmodel', default=False, action="store_true")
    arg_parser.add_argument("--loaddata", default=False, action="store_true")
    arg_parser.add_argument("--amenLen", default=100, type=int)
    args = arg_parser.parse_args()

    nce_criterion = MILNCELoss()

    text_criterion = nn.CrossEntropyLoss()
    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=0)
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    device = args.device
    PATH = args.path
    
    X_train, Xb_train, Xa_train, Xn_train, X_input_train, X_len_train, X_id_train, y_train, X_test, Xb_test, Xa_test, Xn_test, X_input_test, X_len_test, X_id_test, y_test, X_val, Xb_val, Xa_val, Xn_val, X_input_val, X_len_val, X_id_val, y_val = loaddata(load=args.loaddata, amenity_len=args.amenLen)


    X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    Xb_all = np.concatenate((Xb_train, Xb_val, Xb_test), axis=0)
    Xa_all = np.concatenate((Xa_train, Xa_val, Xa_test), axis=0)
    Xn_all = np.concatenate((Xn_train, Xn_val, Xn_test), axis=0)
    X_input_all = np.concatenate((X_input_train, X_input_val, X_input_test), axis=0)
    X_len_all = np.concatenate((X_len_train, X_len_val, X_len_test), axis=0)
    X_id_all = np.concatenate((X_id_train, X_id_val, X_id_test), axis=0)
    y_all = np.concatenate((y_train, y_val, y_test), axis=0)

    X_train, Xb_train, Xa_train, Xn_train, X_input_train, X_len_train, X_id_train, y_train = X_all, Xb_all, Xa_all, Xn_all, X_input_all, X_len_all, X_id_all, y_all
    

    max_y, min_y = 300, 30
    alpha = 10
    if args.mode == 'train':
        model = MODEL(64)
        if args.loadmodel:
            model.load_state_dict(torch.load(args.path))
        model.to(device)
        min_loss = 100000000
        optimizer = Adam(model.parameters(), lr=0.0001)
        for epoch in range(EPOCH):
            #manual shuffle
            n_sample = len(y_train)
            ind = np.arange(n_sample)
            np.random.seed()
            np.random.shuffle(ind)
            X_train = X_train[ind]
            y_train = y_train[ind]
            Xb_train = Xb_train[ind]
            Xn_train = Xn_train[ind]
            X_input_train = X_input_train[ind]
            X_len_train = X_len_train[ind]
            Xa_train = Xa_train[ind]
            train(X_train, Xb_train, Xa_train, Xn_train, X_input_train, X_len_train, y_train)
            
            if (epoch+1)%5==0:
                test(X_test, Xb_test, Xa_test, Xn_test, X_input_test, X_len_test, y_test, 'Test')
                loss, rmse, _, _ = test(X_val, Xb_val, Xa_val, Xn_val, X_input_val, X_len_val, y_val, 'Validation')
                if True or loss < min_loss:
                    min_loss = loss
                    torch.save(model.state_dict(), args.path)
                    print('Model saved!') 
 
    elif args.mode == 'infer':
        model = MODEL(64)
        if args.device == 'cpu':
            model.load_state_dict(torch.load(args.path,  map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(args.path))
        model.to(device)

        test(X_train, Xb_train, Xa_train, Xn_train, X_input_train, X_len_train, y_train, 'Train')
        EBD1, EBD2 = test_emb(X_train, Xb_train, Xa_train, Xn_train, X_input_train, X_len_train, y_train)
        
        metrics = compute_metrics(np.dot(EBD1, EBD2.T))
        print(metrics)
        metrics = compute_metrics(np.dot(EBD2, EBD1.T))
        print(metrics)
        SVM(EBD1, y_train, des='text')
        SVM(EBD2, y_train, des='multimodal')
      
   
