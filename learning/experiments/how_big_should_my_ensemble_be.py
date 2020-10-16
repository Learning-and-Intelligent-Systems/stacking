from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.dropout_gn import DropoutFCGN
from learning.gn import FCGN
from learning.train_graph_net import load_dataset, train, test
from learning.active_train import bald, mc_dropout_score


class Ensemblify(nn.Module):
	""" Turn your boring nn.Module into an exciting ensemble!
	"""
	def __init__(self, model_class, model_args, ensemble_size=10):
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
    ensemble_model = Ensemblify(FCGN, (14, 64))

    if torch.cuda.is_available():
        model = model.cuda()

    # load the datasets
    # datasets are actually a List(TensorDataset), one for each tower size.
    # differing tower sizes requires tensors of differing dimensions
    train_datasets = load_dataset('random_blocks_(x2000)_5blocks_all.pkl')
    test_datasets = load_dataset('random_blocks_(x2000)_5blocks_all.pkl')

    # train the models
    dropout_accs = train(dropout_model, train_datasets, test_datasets, epochs=3)
    ensemble_accs = train(ensemble_model, train_datasets, test_datasets, epochs=3, is_ensemble=True)
    plt.plot(dropout_accs, label="Dropout", alpha=0.5)
    plt.plot(ensemble_accs, label="Ensemble", alpha=0.5)
    plt.title('Training Accuracies')
    plt.legend()
    plt.show()

    