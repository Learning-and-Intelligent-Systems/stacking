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

def active_train(ensemble, dataloader, data_gen_fn, data_label_fn, logger, args):
    """ Main training function 
    :param args: Commandline arguments such as the number of acquisition points.
    :return: Ensembles. The fully trained ensembles.
    """
    # Initilize dataset.
    

    for tx in range(args.max_acquisitions):
        logger.save_dataset(dataset, tx)

        # Initialize and train models.
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True) 
        
        # TODO: Need to somehow reinitialize the ensemble.
        for model in ensemble:
            train(loader, loader, model, args.n_epochs)
        
        logger.save_ensemble(ensemble, tx)

        # Collect new samples.
        new_xs, new_ys, all_samples = acquire_datapoints(ensemble, args.n_samples, args.n_acquire, args.strategy)
        logger.save_acquisition_data(new_xs, new_ys, all_samples, tx)

        # Add to dataset.
        dataset = add_to_dataset(dataset, new_xs, new_ys)
