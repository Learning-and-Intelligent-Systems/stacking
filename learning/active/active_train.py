import argparse
import numpy as np

from torch.utils.data import DataLoader

from learning.active.acquire import acquire_datapoints
from learning.active.mlp import MLP
from learning.active.toy_data import ToyDataset, ToyDataGenerator
from learning.active.train import train
from learning.active.utils import ActiveExperimentLogger


def add_to_dataset(dataset, new_xs, new_ys):
    """ Create a new dataset by adding the new points to the current data.
    :param dataset: The existing ToyDataset object.
    :param new_xs: (n_acquire, 2) 
    :param new_ys: (n_acquire,)
    :return: A new ToyDataset instance with all datapoints.
    """
    xs = np.concatenate([dataset.xs, new_xs], axis=0)
    ys = np.concatenate([dataset.ys, new_ys], axis=0)
    return ToyDataset(xs, ys)

def active_train(args):
    """ Main training function 
    :param args: Commandline arguments such as the number of acquisition points.
    :return: Ensembles. The fully trained ensembles.
    """
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # Initilize dataset.
    gen = ToyDataGenerator()
    xs, ys = gen.generate_uniform_dataset(N=args.n_train_init)
    dataset = ToyDataset(xs, ys)

    for tx in range(args.max_acquisitions):
        logger.save_dataset(dataset, tx)

        # Initialize and train models.
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True) 
        
        ensemble = [MLP(args.n_hidden, args.dropout) for _ in range(args.n_models)]
        for model in ensemble:
            train(loader, loader, model, args.n_epochs)
        
        logger.save_ensemble(ensemble, tx)

        # Collect new samples.
        new_xs, new_ys, all_samples = acquire_datapoints(ensemble, args.n_samples, args.n_acquire)
        logger.save_acquisition_data(new_xs, new_ys, all_samples, tx)

        # Add to dataset.
        dataset = add_to_dataset(dataset, new_xs, new_ys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-acquisitions', 
                        type=int, 
                        default=200,
                        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-models', type=int, default=5, help='Number of models in the ensemble.')
    parser.add_argument('--n-hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--n-epochs', type=int, default=300)
    parser.add_argument('--n-train-init', type=int, default=100)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--n-acquire', type=int, default=10)
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')

    
    args = parser.parse_args()

    active_train(args)
