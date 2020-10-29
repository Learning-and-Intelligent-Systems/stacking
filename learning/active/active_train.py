import argparse
import numpy as np

from learning.active.acquire import acquire_datapoints
from learning.active.train import train
from learning.active.utils import ActiveExperimentLogger


def active_train(ensemble, dataset, dataloader, data_sampler_fn, data_label_fn, data_pred_fn, data_subset_fn, logger, args):
    """ Main training function 
    :param ensemble: learning.models.Ensemble object to be trained.
    :param dataset: Object containing the data to iterate over. Can be added to.
    :param dataloader: The dataloader linked to the given dataset.
    :param data_sampler_fn:
    :param data_label_fn:
    :param data_pred_fn:
    :param data_subset_fn:
    :param logger: Object used to keep track of training artifacts.
    :param args: Commandline arguments such as the number of acquisition points.
    :return: The fully trained ensemble.
    """
    for tx in range(args.max_acquisitions):
        logger.save_dataset(dataset, tx)

        # Initialize and train models.
        ensemble.reset()
        for model in ensemble.models:
            train(dataloader, dataloader, model, args.n_epochs)
        
        logger.save_ensemble(ensemble, tx)

        # Collect new samples.
        new_xs, new_ys, all_samples = acquire_datapoints(ensemble=ensemble, 
                                                         n_samples=args.n_samples, 
                                                         n_acquire=args.n_acquire, 
                                                         strategy=args.strategy,
                                                         data_sampler_fn=data_sampler_fn,
                                                         data_subset_fn=data_subset_fn,
                                                         data_label_fn=data_label_fn,
                                                         data_pred_fn=data_pred_fn)
        logger.save_acquisition_data(new_xs, new_ys, all_samples, tx)

        # Add to dataset.
        dataset.add_to_dataset(new_xs, new_ys)
