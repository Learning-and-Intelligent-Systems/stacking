import torch
from torch import nn
from torch.nn import functional as F

from learning.utils import View

class TowerLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, visual=False, image_dim=None):
        """ This network is given input of size (N, K, n_in) where N, K can vary per batch.
        :param n_in: Number of block-specific parameters.
        :param n_hidden: Number of hidden units unsed throughout the network.
        :param visual: True is inputs are images
        :param image_dim: width/height (square images) number of pixels (only used 
                            if visual==True)
        """
        super(TowerLSTM, self).__init__()

        self.visual = visual
        if self.visual:
            self.image_dim = image_dim
            kernel_size = 6
            stride = 3
            def calc_fc_size():
                W = (self.image_dim-kernel_size)+1
                W = ((W-kernel_size)/stride)+1
                W = (W-kernel_size)+1
                return int(W)
            fc_layer_size = calc_fc_size()
            self.encoder = nn.Sequential(
                            nn.Conv2d(in_channels=1,
                                            out_channels=n_hidden,
                                            kernel_size=kernel_size),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=kernel_size, 
                                            stride=stride),
                            nn.Conv2d(in_channels=n_hidden,
                                            out_channels=1,
                                            kernel_size=kernel_size),
                            nn.ReLU(),
                            View((-1, fc_layer_size**2)),
                            nn.Linear(fc_layer_size**2, 
                                            n_hidden))                
            gru_input_size = n_hidden
        else:
            gru_input_size = n_in
            
        self.lstm = nn.GRU(input_size=gru_input_size,
                            hidden_size=n_hidden,
                            num_layers=2,
                            batch_first=True)

        self.O = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, 1))
        
        self.n_in, self.n_hidden = n_in, n_hidden

    def forward(self, towers, k):
        """
        :param towers: (N, K, n_in) tensor describing the tower.
        :param k: Number of times to iterate the graph update.
        """
        N, K = towers.shape[:2]
        
        if self.visual:
            x = self.encoder(towers.view(N*K, 1, self.image_dim, self.image_dim)).view(N, K, -1)
            #t = torch.cuda.get_device_properties(0).total_memory
            #c = torch.cuda.memory_cached(0)
            #a = torch.cuda.memory_allocated(0)
            #f = c-a  # free inside cache
            #print('Total memory, cached, allocated [GiB]:', t/1073741824, c/1073741824, a/1073741824)
            lstm_input = x
        else:
            lstm_input = towers
        
        x = torch.flip(lstm_input, dims=[1])
            
        x, _ = self.lstm(x)

        #x = torch.mean(x, dim=1)
        x = torch.sigmoid(self.O(x.reshape(-1, self.n_hidden)).view(N, K))
        return x.prod(dim=1)

        
