import torch
from torch.utils.data import DataLoader, Dataset

from sleep_classification.sleepstage import print_n_samples_each_class

import os
import numpy as np
import re

def get_balance_class_oversample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y

def _load_npz_file(npz_file):
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

def _load_npz_list_files(npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = _load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

def load_dataloader_for_featureNet(config=None):
    if config == None:
        from argparse import Namespace
        config = {
            'data_dir': 'G:/내 드라이브/EEG_classification/output',
            'fold_idx': 2,
            'n_fold': 20,
            'train_ratio':0.9,
            'batch_size':512,
        }
        config = Namespace(**config)

    allfiles = os.listdir(config.data_dir)
    npzfiles = []

    for idx, f in enumerate(allfiles):
        if ".npz" in f:
            npzfiles.append(os.path.join(config.data_dir, f))
    npzfiles.sort()

    var_files = np.array_split(npzfiles, config.n_fold)
    train_files = np.setdiff1d(npzfiles, var_files[config.fold_idx])



    print('train_file = {} subject_file = {}'.format(len(train_files),len(var_files[config.fold_idx])))

    print("\n========== [Fold-{}] ==========\n".format(config.fold_idx))
    print("Load training set:")
    train_data, train_label = _load_npz_list_files(npz_files=train_files)
    print(" ")
    print("Load validation set:")
    test_data, test_label = _load_npz_list_files(npz_files=var_files[config.fold_idx])
    print(" ")

    train_data = np.squeeze(train_data)
    test_data = np.squeeze(test_data)
    train_data = train_data[:,np.newaxis,:]
    test_data = test_data[:,np.newaxis,:]

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)  

    print("Training set: {}, {}".format(train_data.shape, train_label.shape))
    print_n_samples_each_class(train_label)
    print(" ")
    print("test set: {}, {}".format(test_data.shape, test_label.shape))
    print_n_samples_each_class(test_label)
    print(" ")

    print(train_data.shape[0])
    train_cnt = int(train_data.shape[0] * config.train_ratio)
    valid_cnt = train_data.shape[0] - train_cnt

    print("train {} valid {}".format(train_cnt,valid_cnt))

    indices = torch.randperm(train_data.shape[0])
    train_x,valid_x = torch.index_select(train_data,dim =0,index= indices).split([train_cnt,valid_cnt],dim = 0)
    train_y,valid_y = torch.index_select(train_label,dim =0,index= indices).split([train_cnt,valid_cnt],dim = 0)
    
    print("train_x {} valid_x {}".format(train_x.shape,valid_x.shape))
    train_dataset = EEGdataset(data=train_x,label=train_y)
    valid_datset = EEGdataset(data=valid_x,label=valid_y)
    test_dataset = EEGdataset(data=test_data,label=test_label)

    print("train {} valid {} test {}".format(len(train_dataset),len(valid_datset),len(test_dataset)))

    train_loader = DataLoader(dataset=train_dataset,batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_datset,batch_size=config.batch_size,shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,batch_size=config.batch_size,shuffle=False)
    return train_loader, valid_loader, test_loader

class EEGdataset(Dataset):
    def __init__(
        self,
        data,
        label
        ):
        self.data = data
        self.label = label

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        x = self.data[idx]
        y = self.label[idx].long()        
        return x,y
    
    