import os
import json
import time
import re
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
import pandas as pd
import codecs
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel

def loaddata(load=False, amenity_len=100):

  
    train_file = 'data/data_train.npz'
    test_file = 'data/data_test.npz'
    val_file = 'data/data_val.npz'
    if load==True:
        dat = np.load(train_file)
        X_train, Xb_train, Xa_train, Xn_train, X_input_train, X_len_train, X_id_train, y_train = dat['X_train'], dat['Xb_train'], dat['Xa_train'], dat['Xn_train'], dat['X_input_train'], dat['X_len_train'], dat['X_id_train'], dat['y_train']
        dat = np.load(test_file)
        X_test, Xb_test, Xa_test, Xn_test, X_input_test, X_len_test, X_id_test, y_test = dat['X_test'], dat['Xb_test'], dat['Xa_test'], dat['Xn_test'], dat['X_input_test'], dat['X_len_test'], dat['X_id_test'], dat['y_test']
        dat = np.load(val_file)
        X_val, Xb_val, Xa_val, Xn_val, X_input_val, X_len_val, X_id_val, y_val = dat['X_val'], dat['Xb_val'], dat['Xa_val'], dat['Xn_val'], dat['X_input_val'], dat['X_len_val'], dat['X_id_val'], dat['y_val']
    return  X_train, Xb_train, Xa_train, Xn_train, X_input_train, X_len_train, X_id_train, y_train, X_test, Xb_test, Xa_test, Xn_test, X_input_test, X_len_test, X_id_test, y_test, X_val, Xb_val, Xa_val, Xn_val, X_input_val, X_len_val, X_id_val, y_val

if __name__ == '__main__':
    loaddata(load=True)

