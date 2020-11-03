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
        '''
        def calc_fc_size():
            W = ((image_dim-11)/4)+1
            W = ((W-3)/2)+1
            W = (W-5)+1
            W = ((W-3)/2)+1
            W = (W-3)+1
            W = (W-3)+1
            W = (W-3)+1
            return int(W)
        '''

        self.encoder = nn.Sequential(
                        nn.Conv2d(2, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
                        nn.ReLU(True),
                        nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
                        nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
                        nn.ReLU(True),
                        nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
                        nn.Conv2d(8, 1, 3, stride=2, padding=1),  # b, 8, 3, 3
                        nn.ReLU(True),
                    )
                                       
        self.output = nn.Sequential(
                        View((-1, image_dim**2)),
                        nn.Linear(image_dim**2, 
                                        16),
                        nn.ReLU(),
                        nn.Linear(16, 
                                    1),
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
                h_small_im = torchvision.transforms.ToPILImage()(h_small[n,:,:].cpu())
                h[n,:,:,:] = torchvision.transforms.ToTensor()(h_small_im.resize((image_dim, image_dim)))

            if torch.cuda.is_available():
                h = h.cuda()

        # TODO: try later to map all hidden states to a stability prediction and take the .prod()
        y = self.output(h)
        return y
