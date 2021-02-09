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

def load_dataloader_for_featureNet(data_dir,fold_idx,batch_size):
    allfiles = os.listdir(data_dir)
    npzfiles = []
    for idx, f in enumerate(allfiles):
        if ".npz" in f:
            npzfiles.append(os.path.join(data_dir, f))
    npzfiles.sort()

    subject_files = []
    for idx, f in enumerate(allfiles):
        if fold_idx < 10:
            pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(fold_idx))
        else:
            pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(fold_idx))
        if pattern.match(f):
            subject_files.append(os.path.join(data_dir, f))

    if len(subject_files) == 0:
        for idx, f in enumerate(allfiles):
            if fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(fold_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(data_dir, f))

    train_files = list(set(npzfiles) - set(subject_files))
    train_files.sort()
    subject_files.sort()

    print('train_file = {} subject_file = {}'.format(len(train_files),len(subject_files)))

    print("\n========== [Fold-{}] ==========\n".format(fold_idx))
    print("Load training set:")
    data_train, label_train = _load_npz_list_files(npz_files=train_files)
    print(" ")
    print("Load validation set:")
    data_val, label_val = _load_npz_list_files(npz_files=subject_files)
    print(" ")

    # Reshape the data to match the input of the model - conv2d
    data_train = np.squeeze(data_train)
    data_val = np.squeeze(data_val)
    data_train = data_train[:,np.newaxis,:]
    data_val = data_val[:,np.newaxis,:]

    data_train = data_train.astype(np.float32)
    label_train = label_train.astype(np.int32)
    data_val = data_val.astype(np.float32)
    label_val = label_val.astype(np.int32)  

    print("Training set: {}, {}".format(data_train.shape, label_train.shape))
    print_n_samples_each_class(label_train)
    print(" ")
    print("Validation set: {}, {}".format(data_val.shape, label_val.shape))
    print_n_samples_each_class(label_val)
    print(" ")

    # Use balanced-class, oversample training set
    x_train, y_train = get_balance_class_oversample(
        x=data_train, y=label_train
    )
    print("Oversampled training set: {}, {}".format(
        x_train.shape, y_train.shape
    ))

    print_n_samples_each_class(y_train)

    print(" ")
    train_loader = DataLoader(dataset=EEGdataset(data=x_train,label=y_train),batch_size=batch_size,shuffle=True)
    valid_loader = DataLoader(dataset=EEGdataset(data=data_val,label=label_val),batch_size=batch_size,shuffle=False) 
    return train_loader, valid_loader

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
        y = self.labels[idx]        
        return x,y
    
    