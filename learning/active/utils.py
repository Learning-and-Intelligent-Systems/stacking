import os
import pickle
import time
import torch

from torch.utils.data import DataLoader

from learning.active.mlp import MLP


class ExperimentLogger:

    def __init__(self, exp_path):
        self.exp_path = exp_path

        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            self.args = pickle.load(handle)

    @staticmethod
    def setup_experiment_directory(args):
        """
        Setup the directory structure to store models, figures, datasets
        and parameters relating to an experiment.
        """
        root = 'learning/active/experiments'
        exp_name = args.exp_name if len(args.exp_name) > 0 else 'exp'
        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_dir = '%s-%s' % (exp_name, ts)
        exp_path = os.path.join(root, exp_dir)

        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, 'figures'))
        os.mkdir(os.path.join(exp_path, 'models'))
        os.mkdir(os.path.join(exp_path, 'datasets'))

        with open(os.path.join(exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return ExperimentLogger(exp_path)

    def save_dataset(self, dataset, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'wb') as handle:
            pickle.dump(dataset, handle)

    def load_dataset(self, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset

    def get_figure_path(self, fname):
        return os.path.join(self.exp_path, 'figures', fname)

    def save_model(self, model, fname):
        torch.save(model.state_dict(), os.path.join(self.exp_path, 'models', fname))

    def load_model(self, fname):
        model = MLP(n_hidden=self.args.n_hidden, dropout=self.args.dropout)
        model.load_state_dict(torch.load(os.path.join(self.exp_path, 'models', fname)))
        return model

    def get_ensemble(self):
        ensemble = []
        for mx in range(0, self.args.n_models):
            ensemble.append(self.load_model('net_%d.pt' % mx))
        return ensemble

class ActiveExperimentLogger:

    def __init__(self, exp_path):
        self.exp_path = exp_path

        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            self.args = pickle.load(handle)

    @staticmethod
    def setup_experiment_directory(args):
        """
        Setup the directory structure to store models, figures, datasets
        and parameters relating to an experiment.
        """
        root = 'learning/active/experiments'
        exp_name = args.exp_name if len(args.exp_name) > 0 else 'exp'
        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_dir = '%s-%s' % (exp_name, ts)
        exp_path = os.path.join(root, exp_dir)

        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, 'figures'))
        os.mkdir(os.path.join(exp_path, 'models'))
        os.mkdir(os.path.join(exp_path, 'datasets'))
        os.mkdir(os.path.join(exp_path, 'acquisition_data'))

        with open(os.path.join(exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return ActiveExperimentLogger(exp_path)

    def save_dataset(self, dataset, tx):
        fname = 'active_%d.pkl' % tx
        with open(os.path.join(self.exp_path, 'datasets', fname), 'wb') as handle:
            pickle.dump(dataset, handle)

    def load_dataset(self, tx):
        fname = 'active_%d.pkl' % tx
        with open(os.path.join(self.exp_path, 'datasets', fname), 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset

    def get_figure_path(self, fname):
        return os.path.join(self.exp_path, 'figures', fname)

    def load_model(self, path):
        model = MLP(n_hidden=self.args.n_hidden, dropout=self.args.dropout)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model

    def get_ensemble(self, tx):
        ensemble = []
        for mx in range(0, self.args.n_models):
            path = os.path.join(self.exp_path, 'models', str(tx), 'net_%d.pt' % mx)
            ensemble.append(self.load_model(path))
        return ensemble

    def save_ensemble(self, ensemble, tx):
        os.mkdir(os.path.join(self.exp_path, 'models', str(tx)))
        for mx, model in enumerate(ensemble):
            path = os.path.join(self.exp_path, 'models', str(tx), 'net_%d.pt' % mx)
            torch.save(model.state_dict(), os.path.join(path))

    def save_acquisition_data(self, new_xs, new_ys, samples, tx):
        data = {
            'acquired_xs': new_xs,
            'acquired_ys': new_ys,
            'samples': samples
        }
        path = os.path.join(self.exp_path, 'acquisition_data', 'acquired_%d.pkl' % tx)
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)

    def load_acquisition_data(self, tx):
        path = os.path.join(self.exp_path, 'acquisition_data', 'acquired_%d.pkl' % tx)
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data['acquired_xs'], data['acquired_ys'], data['samples']



def get_predictions(dataset, models):
    """
    :param dataset: A ToyDataset object with N examples.
    :param models: A list of K models on which to get predictions.
    :return: A (NxK) array with classification probabilities for each model.
    """
    loader = DataLoader(dataset, shuffle=False, batch_size=32)

    model_predictions = []
    for model in models:
        preds = []
        for tensor, _ in loader:
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                preds.append(model.forward(tensor))
            
        preds = torch.cat(preds, dim=0)
        model_predictions.append(preds)

    return torch.cat(model_predictions, dim=1)
