import torch, math
import torch.nn as nn
import numpy as np

r"""
    We define a layer taking care that we recieve as input the adjacency matrix and 
    the node atributes as a 3 dimensional tensor (the data batched). Moreover, this 
    layer is defined for complex attributes. 
"""


class MagNet_layer(nn.Module):

    def __init__(self, 
                 in_features, 
                 out_features, 
                 device, 
                 bias = True, 
                 K=1, 
                 simetric = True, 
                 Matrix = 'Laplacian_N', 
                 q = 0.25):

        r"""
        To define this layer we use 
            Parameteers:
        
        :param: in_features -> The 3-dimension of the tensor *node features*
        :param: out_features-> The 3-dimension of the output tensor *node features* 
        :param: device      -> Where we compute the calculations
        :param: bias        -> We add a bias factor
        :param: K           -> The order of the polinomial
        :param: simetric    -> If K = 1 and c0 = -c1
        :param: Matrix      -> The GSO we use to define
        """
        super(MagNet_layer, self).__init__()
        self.order = K
        self.simetric = simetric
        self.in_features = in_features
        self.out_features = out_features
        self.q = q


        if K == 1 and simetric:
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        else:
            self.weight = nn.Parameter(torch.FloatTensor(K+1, in_features, out_features)).to(device)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        
    def reset_parameters(self):
        '''
        Escala els paràmetres, ja que al crearlos aleatoris, no volem 
        valors massa disparats.
        '''
        
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, L):
        '''
        Define the matrix Asked

        Now we use by defect l_max ~ 2
        '''
        # Define the Magnetic Laplacian Normalized
        sizes = L.size()
        I = torch.stack([torch.eye(sizes[1]) for _ in range(sizes[0])])


        # Scale the magnetic laplacian with l_max ~ 2


        if self.order == 1 and self.simetric:

            X = X.reshape(X.size()[0]*X.size()[1], X.size()[2])
            X = torch.mm(X.real, self.weight) + 1.0j*torch.mm(X.imag, self.weight)

            X = X.reshape(L.size()[0], L.size()[1], self.weight.size()[-1])
            H = I - L + I*1.0j
            output = (torch.bmm(H.real, X.real) - torch.bmm(H.imag,X.imag)+
                      1.0j*(torch.bmm(H.imag,X.real)+torch.bmm(H.real,X.imag))) 
            
            if self.bias is not None:
                return output + self.bias
            else:
                return output
            

    def __repr__(self):
        return self.__class__.__name__ + '(' \
                    + str(self.in_features) \
                    + str(self.out_features) + ')'

class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real, img):
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real, img=None):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img
