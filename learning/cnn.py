import torch
from torch import nn
from torch.nn import functional as F

from learning.utils import View

class TowerCNN(nn.Module):
    def __init__(self, image_dim, n_hidden=32):
        """ This network is given input of size (N, K, n_in) where N, K can vary per batch.
        :param image_dim: width/height (square images) number of pixels (only used 
                            if visual==True)
        """
        super(TowerCNN, self).__init__()

        kernel_size = 9
        stride = 3
        def calc_fc_size():
            W = (image_dim-kernel_size)+1
            W = ((W-kernel_size)/stride)+1
            W = (W-kernel_size)+1
            return int(W)
        fc_layer_size = calc_fc_size()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels=2,
                                        out_channels=n_hidden,
                                        kernel_size=kernel_size),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=kernel_size, 
                                        stride=stride),
                        nn.Conv2d(in_channels=n_hidden,
                                        out_channels=n_hidden,
                                        kernel_size=kernel_size),
                        nn.ReLU(),
                        View((-1, n_hidden*fc_layer_size**2)),
                        nn.Linear(n_hidden*fc_layer_size**2, 
                                        n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, 1),
                        nn.Sigmoid())

    def forward(self, images, k=None):
        """
        :param images: (N, K, image_dim, image_dim) tensor describing the tower in images.
        """
        x = self.encoder(images)
        return x

        
