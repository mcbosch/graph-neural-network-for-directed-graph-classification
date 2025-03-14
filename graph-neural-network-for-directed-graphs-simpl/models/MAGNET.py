import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_cheb import MagNet_layer
from readouts.basic_readout import readout_function



class MagNet(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device):
        super(MagNet, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout
        
        # Graph convolution layer
        self.layers = []
        for i in range(n_layer):
           if i == 0:
             self.layers.append(MagNet_layer(n_feat, agg_hidden, device))
           else:
             self.layers.append(MagNet_layer(agg_hidden, agg_hidden, device))
        
        # Fully-connected layer
        self.fc1 = nn.Linear(2*agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)

    @staticmethod
    def complex_relU(X): return X.detach().apply_(lambda z: z if z.real >= 0 else 0)
    
    def forward(self, data):
        x, L = data[:2]
        sizes = x.size()
        x = x + torch.zeros(sizes[0],sizes[1],sizes[2])*1.0j
        
        for i in range(self.n_layer):
           # Graph convolution layer

           x = MagNet.complex_relU(self.layers[i](x, L))

           # Dropout ULTIMA CAPA
           if i != self.n_layer - 1:
             
             x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Readout
        x = readout_function(x, self.readout)
        
        # Fully- layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x
        
    def __repr__(self):
        layers = ''
        
        for i in range(self.n_layer):
            layers += str(self.layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n' + 'USING MAGNET' + '\n'
        return layers
    
    def help(self):
       print('The function help for an explanation is not defined\n if u want an explanation please go to the source code')
       

