import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolution_Block(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kenel_size = kernel_size
        self.padding = padding

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=self.kenel_size,padding=self.padding),
            nn.BatchNorm1d(num_features=self.out_channel),
            nn.ReLU()
        )

    def forword(self,x):
        return self.layers(x)


class DeepfeatureNet(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes,
        is_train,
        use_dropout,
        sampling_rate = 100
        ):  

        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.is_train = is_train
        self.use_dropout = use_dropout
        self.fs = sampling_rate

        if is_train:
            self.dropout1 = nn.Dropout()
            self.dropout2 = nn.Dropout()

        else:
            self.dropout1 = nn.Dropout(p=0.0)
            self.dropout2 = nn.Dropout(p=0.0)
        
        self.flat = nn.Flatten(start_dim=1)
        #input_size = (batch_size,1,3000)

        #cnn1 
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=64,kernel_size=fs//2,stride=6,padding=22), # need add weight decay = 1e-3
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8,stride=8,padding=2),
            self.dropout1,
            Convolution_Block(in_channel=64,out_channel=128,kernel_size=8,padding = 4),
            Convolution_Block(in_channel=128,out_channel=128,kernel_size=8,padding = 3),
            Convolution_Block(in_channel=128,out_channel=128,kernel_size=8,padding = 4),#(100,128,64)
            nn.MaxPool1d(kernel_size=4,stride=4),#(100,128,16)
            self.flat,
        )        
        
        #cnn2
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,out_channels=64,kernel_size=fs*4,stride=fs//2,padding=175),
            nn.MaxPool1d(kernel_size=4,stride=4),
            self.dropout2,
            Convolution_Block(in_channel=64,out_channel=128,kernel_size=6,padding=3),
            Convolution_Block(in_channel=128,out_channel=128,kernel_size=6,padding=3),
            Convolution_Block(in_channel=128,out_channel=128,kernel_size=6,padding=3),#(100,128,16)
            nn.MaxPool1d(kernel_size=2,stride=2),#(100,128,8)
            self.flat,
        )
        
        self.output_layer = nn.Sequential(
            #after concat
            nn.Linear(in_features=3072,out_features=self.n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self,x):
        small = self.cnn1(x)
        big = self.cnn2(x)
        concated = torch.cat(small,big,dim=1)

        y = self.output_layer(concated)
        return y