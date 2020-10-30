import torch
import torch.nn as nn

from learning.models.mlp_dropout import MLP


class Ensemble(nn.Module):
    """ A helper class to represent a collection of models.

    Intended usage:
    This class is designed to be used for active learning. It only needs
    to be initialized once outside of the active loop. Every time we want 
    new models, we only need to call reset().
    
    To save an ensemble, save it as a regular PyTorch model. Only a single 
    file is needed for the whole ensemble.

    When loading, an ensemble with the same base parameters will need to
    be used. After that the forward function can be used to get predictions
    from all models.
    """
    def __init__(self, base_model, base_args, n_models):
        """ Save relevant information to reinitialize the ensemble later on.
        :param base_model: The class of a single ensemble member.
        :param base_args: The arguments used to initialize the base member.
        :param n_models: The number of models in the ensemble.
        """
        super(Ensemble, self).__init__()
        self.base_model = base_model
        self.base_args = base_args
        self.n_models = n_models
        self.reset()

    def reset(self):
        """ Initialize (or re-initialize) all the models in the ensemble."""
        self.models = nn.ModuleList([self.base_model(**self.base_args) for _ in range(self.n_models)])

    def forward(self, x):
        """ Return a prediction for each model in the ensemble.
        :param x: (N, *) Input tensor compatible with the base_model.
        :return: (N, n_models), class prediction for each model.
        """
        preds = [self.models[ix].forward(x, k=1) for ix in range(self.n_models)]
        return torch.cat(preds, dim=1)


# Test creation of an ensemble.
if __name__ == '__main__':
    ensemble = Ensemble(base_model=MLP, 
                        base_args={'n_hidden': 4, 'dropout': 0.}, 
                        n_models=5)
    # Try saving the model.
    torch.save(ensemble.state_dict(), 'scratch/mlp_ensemble.pt')
    
    print('1st ensemble')
    print(ensemble.models[0].lin1.weight)
    # Reinitialize the model.
    ensemble.reset()

    print('Ensemble 2: After reset')
    print(ensemble.models[0].lin1.weight)

    torch.save(ensemble.state_dict(), 'scratch/mlp_ensemble2.pt')

    # Load the model.
    ensemble1_load = Ensemble(base_model=MLP,
                              base_args={'n_hidden': 4, 'dropout': 0.},
                              n_models=5)
    ensemble1_load.load_state_dict(torch.load('scratch/mlp_ensemble.pt'))

    ensemble2_load = Ensemble(base_model=MLP,
                              base_args={'n_hidden': 4, 'dropout': 0.},
                              n_models=5)
    ensemble2_load.load_state_dict(torch.load('scratch/mlp_ensemble2.pt'))

    print('Loaded 1st ensemble.')
    print(ensemble1_load.models[0].lin1.weight)

    print('Loaded 2nd ensemble.')
    print(ensemble2_load.models[0].lin1.weight)

    ensemble1_load.reset()
    x = torch.ones(10, 2)
    p = ensemble.forward(x)
    print(p.shape)