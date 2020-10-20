import torch
from torch import nn
from torch.nn import functional as F

from learning.utils import View

class TowerConvRNN(nn.Module):
    def __init__(self, image_dim, n_hidden=32):
        """ This network is given input of size (N, K, n_in) where N is the batch size, 
        and K is the number of blocks in the tower. K can vary per batch.
        :param image_dim: width/height (square images) number of pixels
        """
        super(TowerConvRNN, self).__init__()
        # make 
        kernel_size = 9
        stride = 3
        def calc_fc_size():
            W = (image_dim-kernel_size)+1
            W = ((W-kernel_size)/stride)+1
            W = (W-kernel_size)+1
            return int(W)
        self.hidden_dim = calc_fc_size()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels=2,
                                        out_channels=n_hidden,
                                        kernel_size=kernel_size),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=kernel_size, 
                                        stride=stride),
                        nn.Conv2d(in_channels=n_hidden,
                                        out_channels=1,
                                        kernel_size=kernel_size),
                        nn.ReLU())
        
        self.insert_h = int(image_dim/2-self.hidden_dim/2)
                                       
        self.output = nn.Sequential(
                        View((-1, self.hidden_dim**2)),
                        nn.Linear(self.hidden_dim**2, 
                                        n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, 1),
                        nn.Sigmoid())

    def forward(self, images, k=None):
        """
        :param images: (N, K, image_dim, image_dim) tensor describing the tower in images.
        """
        N, K, image_dim, image_dim = images.shape
        h_0 = torch.zeros(N, 1, image_dim, image_dim)
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
        h = h_0
        for k in range(K): 
            # TODO: might need to flip this!
            input = torch.cat([images[:,k,:,:].view(N,1,image_dim, image_dim), h], dim=1)
            h_small = self.encoder(input)
            h = torch.zeros(N, 1, image_dim, image_dim)
            if torch.cuda.is_available():
                h = h.cuda()
            h[:,:,self.insert_h:self.insert_h+self.hidden_dim, self.insert_h:self.insert_h+self.hidden_dim] = h_small

        # TODO: try later to map all hidden states to a stability prediction and take the .prod()
        y = self.output(h_small)
        return y