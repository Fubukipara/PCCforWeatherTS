import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PCC(nn.Module):
    
    def __init__(self, configs):
        super(PCC, self).__init__()
        
        self.hidden_dim = configs.PCC_hidden_dim
        self.act = configs.PCC_activation
        self.dropout = configs.PCC_dropout
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        if self.act == 'relu':
            self.activation = nn.ReLU()
        if self.act == 'gelu':
            self.activation = nn.GELU()
        if self.act == 'tanh':
            self.activation = nn.Tanh()
            
        # Initialize linear layers
        self.MCC_linear1 = nn.Linear(self.enc_in, self.hidden_dim)
        self.MCC_linear2 = nn.Linear(self.hidden_dim, self.enc_in)
        self.SC_linear1 = nn.Linear(self.enc_in, self.hidden_dim)
        self.SC_linear2 = nn.Linear(self.hidden_dim, self.enc_in)

        # Initialize differential matrix
        self.diff_1 = nn.Parameter(torch.ones(self.pred_len,self.hidden_dim))
        self.diff_2 = nn.Parameter(torch.ones(self.pred_len,self.hidden_dim))

            
    def MCC(self,x,reference):

        x = x - reference
        
        x = self.activation(self.MCC_linear1(x))
        
        x = nn.Dropout(p=self.dropout)(x)
        
        x = self.MCC_linear2(x)

        x = x + reference
        
        return x
        
    def SC(self,x):
        
        x = self.activation(self.SC_linear1(x))
        
        x = nn.Dropout(p=self.dropout)(x)

        h = torch.exp(self.diff_1)*x - torch.exp(self.diff_2)*x
        
        x = self.SC_linear2(h)
        
        return x
        
            
            