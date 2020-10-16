import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from learning.active.toy_data import ToyDataset


class MLP(nn.Module):
    def __init__(self, n_hidden, dropout):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.dropout = dropout

        self.lin1 = nn.Linear(2, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_hidden)
        self.lin4 = nn.Linear(n_hidden, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        #self.init_weights()
        self.sample_dropout_masks()

    def init_weights(self):
        for l in [self.lin1, self.lin2, self.lin3, self.lin4]:
            #nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
            #nn.init.normal_(l.weight, std=0.18)
            nn.init.uniform_(l.weight, -1., 1.)
    def sample_dropout_masks(self):
        dropout_probs = torch.empty(1, self.n_hidden).fill_(1 - self.dropout)
        self.mask1 = torch.bernoulli(dropout_probs)
        self.mask2 = torch.bernoulli(dropout_probs)
        self.mask3 = torch.bernoulli(dropout_probs)

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = x*self.mask1
        x = self.relu(self.lin2(x))
        x = x*self.mask2
        x = self.relu(self.lin3(x))
        x = x*self.mask3

        return self.sigmoid(self.lin4(x))



    def plot_decision_boundary(self, resolution, fname, k):
        x1 = torch.arange(-1, 1, resolution)
        x2 = torch.arange(-1, 1, resolution)

        x1s, x2s = torch.meshgrid(x1, x2)
        K = x1s.shape[0]
        x = torch.cat([x1s.reshape(K*K, 1), x2s.reshape(K*K, 1)], dim=1)
        
        dataset = ToyDataset(x, torch.zeros((K*K,)))
        loader = DataLoader(dataset, shuffle=False, batch_size=32)
        
        model_predictions = []
        for kx in range(k):
            self.sample_dropout_masks()
            preds = []
            for tensor, _ in loader:
                with torch.no_grad():
                    preds.append(self.forward(tensor).squeeze())
            
            preds = torch.cat(preds, dim=0)
            model_predictions.append(preds)
            preds = (preds > 0.5).reshape(K, K)
            plt.close()
            plt.pcolormesh(x1s.numpy(), x2s.numpy(), preds.numpy())
            plt.savefig(fname)

        return model_predictions