from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.dropout_gn import DropoutFCGN
from learning.gn import FCGN
from learning.train_graph_net import load_dataset, train, test
from learning.active_train import bernoulli_bald


class Ensemblify(nn.Module):
    """ Turn your boring nn.Module into an exciting ensemble!
    """
    def __init__(self, model_class, model_args, ensemble_size=1000):
        super(Ensemblify, self).__init__()
        # wrap with ModuleList so that we register params for ever model
        self.models = nn.ModuleList([model_class(*model_args) for _ in range(ensemble_size)])
        self.ensemble_size = ensemble_size

    def forward(self, *args, **kwargs):
        """ Forward pass on the ensemble

        Note, this function stacks all the outputs of the models in the
        ensemble into a tensor with the batch size along the first dim
        and the ensemble size along the second dim. In order to do this
        the output of each model in the ensemble must be a tensor of
        the same shape

        Arguments:
            *args {*} -- whatever arguments you want to pass to the models
            *kwargs {**} -- whatever keyword arguments you want to pass to the models

        Returns:
            torch.Tensor -- [batch_size x ensemble_size x ...]
        """
        ys = [m(*args, **kwargs) for m in self.models]
        return torch.stack(ys, axis=1)


if __name__ == '__main__':
    dropout_model = DropoutFCGN(14, 64)
    ensemble_model = Ensemblify(FCGN, (14, 64), ensemble_size=150)

    if torch.cuda.is_available():
        model = model.cuda()

    # load the datasets
    # datasets are actually a List(TensorDataset), one for each tower size.
    # differing tower sizes requires tensors of differing dimensions
    train_datasets = load_dataset('random_blocks_(x40000)_5blocks_all.pkl')
    test_datasets = load_dataset('random_blocks_(x2000)_5blocks_all.pkl')

    # train the models
    dropout_accs = train(dropout_model, train_datasets, test_datasets, epochs=2)
    ensemble_accs = train(ensemble_model, train_datasets, test_datasets, epochs=2, is_ensemble=True)
    plt.plot(dropout_accs, label="Dropout", alpha=0.5)
    plt.plot(ensemble_accs, label="Ensemble", alpha=0.5)
    plt.title('Training Accuracies')
    plt.legend()
    plt.savefig('train_accuracies.png')
    plt.clf()

    # sample predictions from both the dropout and ensemble models on the whole
    # test dataset
    N = ensemble_model.ensemble_size - 1 # number of dropout samples
    M = 100 # number of subset samples to calculate variance
    dropout_model.train(True) # make sure dropout is enabled

    ensemble_preds = []
    dropout_preds = []
    for dataset in test_datasets:
        # pull out the input tensors for the whole dataset
        towers = dataset[:][0]
        if torch.cuda.is_available():
            towers = towers.cuda()
        # run the model on everything
        with torch.no_grad():
            ensemble_preds.append(ensemble_model.forward(towers, k=towers.shape[1]-1))
            dropout_preds.append(dropout_model.sample(towers, k=towers.shape[1]-1, num_samples=ensemble_model.ensemble_size))

    ensemble_preds = torch.cat(ensemble_preds, axis=0)
    dropout_preds = torch.cat(dropout_preds, axis=0)

    # and compute the BALD score using subsets of those samples to see how
    # sample size effects convergence
    ensemble_scores = []
    ensemble_vars = []
    dropout_scores = []
    dropout_vars = []
    for n in range(1, N):
        ensemble_bald = torch.zeros(7000, M)
        dropout_bald = torch.zeros(7000, M)
        for m in range(M):
            idxs = np.random.choice(N, n, replace=False)

            ensemble_bald[:,m] = bernoulli_bald(ensemble_preds[:,idxs])
            dropout_bald[:,m] = bernoulli_bald(dropout_preds[:,idxs])

        # # build an array of subset idxs
        # idxs = np.zeros([M,n])
        # for m in range(M):
        #     idxs[m] = np.random.choice(N, n, replace=False)


        # ensemble_bald = bernoulli_bald(ensemble_preds[:,idxs].view(-1,n)).view(-1, M)
        # dropout_bald = bernoulli_bald(dropout_preds[:,idxs].view(-1,n)).view(-1, M)

        # print(ensemble_bald.shape, dropout_bald.shape)

        ensemble_scores.append(ensemble_bald.mean().item())
        dropout_scores.append(dropout_bald.mean().item())
        ensemble_vars.append(torch.var(ensemble_bald, axis=1).mean().item())
        dropout_vars.append(torch.var(dropout_bald, axis=1).mean().item())

    ensemble_scores = np.array(ensemble_scores)
    ensemble_vars = np.array(ensemble_vars)
    dropout_scores = np.array(dropout_scores)
    dropout_vars = np.array(dropout_vars)

    plt.plot(dropout_scores, label='Dropout', color=(1,0,0), alpha=0.5)
    plt.fill_between(x=np.arange(len(dropout_scores)), y1=dropout_scores-dropout_vars, y2=dropout_scores+dropout_vars,
        edgecolor=(1,0,0), facecolor=(1,0,0), alpha=0.2)
    plt.plot(ensemble_scores, label='Ensemble', color=(0,0,1), alpha=0.5)
    plt.fill_between(x=np.arange(len(ensemble_scores)), y1=ensemble_scores-ensemble_vars, y2=ensemble_scores+ensemble_vars,
        edgecolor=(0,0,1), facecolor=(0,0,1), alpha=0.2)
    plt.legend()
    plt.xlabel('Num Models in Ensemble / Num MC Samples')
    plt.ylabel('Mean BALD score on Pool')
    plt.title('How many models to represent the posterior?')
    plt.show()
