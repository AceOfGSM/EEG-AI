import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Convolution_Block(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride = 1):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kenel_size = kernel_size
        self.stride = stride

        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=self.kenel_size,stride=self.stride),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.out_channel),
        )

    def forward(self,x):
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


class DeepSleepNet(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes,
        is_train,
        use_dropout,
        use_rnn = True,
        sampling_rate = 100):

        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.is_train = is_train
        self.use_dropout = use_dropout
        self.fs = sampling_rate

        if is_train:
            self.dropout = nn.Dropout() 
        else:
            self.dropout = nn.Dropout(p=0.0)

        self.flat = nn.Flatten(start_dim=1)

        #input_size = (batch_size,1,3000)

        self.cnn_model = nn.Sequential(
            Convolution_Block(in_channel=1,out_channel=128,kernel_size=self.fs//2,stride=self.fs//16),
            nn.MaxPool1d(kernel_size=8,stride=8,padding = 2),
            self.dropout,
            Convolution_Block(in_channel=128,out_channel=128,kernel_size=8),
            Convolution_Block(in_channel=128,out_channel=128,kernel_size=8),
            Convolution_Block(in_channel=128,out_channel=128,kernel_size=8),
            nn.MaxPool1d(kernel_size=4,stride=4,padding = 1),
            self.flat,
            self.dropout
        )   

        self.rnn_model = nn.Sequential(
            nn.LSTM(input_size=69,hidden_size = 128,num_layers = 1, batch_first = True,bidirectional = False),
        )

        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(in_features=128,out_features=5),
            nn.Softmax(dim = -1)    
        )
    
    def forward(self,x):
        #|x| = (bs,1,3000)
        cnn = self.cnn_model(x)
        #|cnn| = (bs,2048)
        cnn = F.pad(cnn,pad = (0,22)).view(cnn.size(0),30,-1)
        #|cnn| = (bs,30_sequnce_length,input_dims)
        rnn, _ = self.rnn_model(cnn)
        #|rnn| = ([bs, 30_sequence_length, 128_hidden_size])
        rnn = rnn[:,-1]
        #|rnn| = (bs,128)
        y = self.fc(rnn)
        #|y| = (bs,5)   
        return y

class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        size, kernel, stride = input.size(-1), self.weight.size(2), self.stride[0]
        padding = kernel - stride - size % stride
        while padding < 0:
            padding += stride   
        if padding != 0:
            input = F.pad(input, (padding // 2, padding - padding // 2))
        return F.conv1d(input=input,
                        weight=self.weight,
                        bias=self.bias,
                        stride=stride,
                        dilation=1,
                        groups=1)

class MaxPoolSamePadding(nn.Module):
    def __init__(self,input_channels,kernel_size,stride):
        super().__init__()
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.MaxPool1d = nn.MaxPool1d(kernel_size=self.kernel_size,stride=self.stride)

    def forward(self, input):
        padding = self.kernel_size - self.stride - self.input_channels % self.stride
        while padding < 0:
            padding += self.stride
        if padding != 0:
            input = F.pad(input, (padding - padding // 2 ,padding//2))
        return self.MaxPool1d(input)