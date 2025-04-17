
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
#import time
# from utils import *
# from huggingface_hub import PyTorchModelHubMixin

## check available device
# device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

class ml_reg(nn.Module):
    def __init__(self,  input_dim,num_classes, bias_init=None,dropout=0.2,n_hiddens=512):
        super(ml_reg, self).__init__()

        self.layer0 = nn.Sequential(
            #nn.Conv1d(in_channels=n_inputs, out_channels=n_hiddens,kernel_size=1, stride=1, bias=True),
            nn.Linear(input_dim,n_hiddens),
            #nn.ReLU(),  ## 2020.03.26: for positive gene expression
            nn.Dropout(dropout)
            )
        
        #self.layer1 = nn.Conv1d(in_channels=n_hiddens, out_channels=n_outputs,kernel_size=1, stride=1, bias=True)
        self.layer1 = nn.Linear(n_hiddens, num_classes)
        self.n_hiddens=n_hiddens
        self.dropout=dropout
        self.bias_init=bias_init
        ## ---- set bias of the last layer ----
        if bias_init is not None:
            self.layer1.bias = bias_init

    ##-------------------------------------
    def forward(self, x):
        
        # print("x.shape - input of forward:", x.shape)    ## [n_tiles,512]
        # print(self.n_hiddens,self.dropout,self.bias_init)
        # exit()
        x = self.layer0(x)
        x = self.layer1(x)

        #print("x.shape - before mean:", x.shape)        ## [n_tiles,512]

        x = torch.mean(x, dim=1)                        ## sum over tiles
        #print("x.shape -- after mean:", x.shape)        ## [n_genes]                     
        # print("x.shape - input of last:", x.shape) 
        # exit()
        return x  
        