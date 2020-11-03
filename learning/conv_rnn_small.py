import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np

from learning.utils import View

class TowerConvRNNSmall(nn.Module):
    def __init__(self, image_dim, n_hidden=8):
        """ This network is given input of size (N, K, n_in) where N is the batch size, 
        and K is the number of blocks in the tower. K can vary per batch.
        :param image_dim: width/height (square images) number of pixels
        """
        super(TowerConvRNNSmall, self).__init__()
        # make 
        kernel_size = 5
        #stride = 3
        def calc_fc_size():
            W = (image_dim-kernel_size)+1
            W = (W-kernel_size)+1
            W = (W-kernel_size)+1
            W = (W-kernel_size)+1
            W = (W-kernel_size)+1
            return int(W)
        self.hidden_dim = calc_fc_size()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels=2,
                                        out_channels=4,
                                        kernel_size=kernel_size),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=4,
                                       out_channels=8,
                                       kernel_size=kernel_size),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=8,
                                        out_channels=16,
                                        kernel_size=kernel_size),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=kernel_size),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32,
                                out_channels=1,
                                kernel_size=kernel_size))
                                       
        self.output = nn.Sequential(
                        View((-1, image_dim**2)),
                        nn.Linear(image_dim**2, 
                                        16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
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
        for k in range(K-1, -1, -1): # start with the top block
            input = torch.cat([images[:,k,:,:].view(N,1,image_dim, image_dim), h], dim=1)
            h_small = self.encoder(input)
            h = torch.zeros(N, 1, image_dim, image_dim)
            # iterate through each hidden state in the batch
            for n in range(N):
                h_small_im = torchvision.transforms.ToPILImage()(h_small[n,:,:])
                h[n,:,:,:] = torchvision.transforms.ToTensor()(h_small_im.resize((image_dim, image_dim)))

            if torch.cuda.is_available():
                h = h.cuda()

        # TODO: try later to map all hidden states to a stability prediction and take the .prod()
        y = self.output(h)
        return y
