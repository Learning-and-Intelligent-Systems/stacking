import argparse
import copy
import numpy as np
import time

from learning.active.acquire import acquire_datapoints
from learning.active.train import train
from learning.active.utils import ActiveExperimentLogger


def split_data(data, n_val):
    """
    Choose n_val of the chosen data points to add to the validation set.
    Return 2 tower_dict structures.
    """
    val_data = copy.deepcopy(data)
    
    total = np.sum([data[k]['towers'].shape[0] for k in data.keys()])
    val_ixs = np.random.choice(np.arange(0, total), 2, replace=False)

    start = 0
    for k in data.keys():
        end = start + data[k]['towers'].shape[0]
        
        tower_ixs = val_ixs[np.logical_and(val_ixs >= start,
                                           val_ixs < end)] - start

        train_mask = np.ones(data[k]['towers'].shape[0], dtype=bool)
        train_mask[tower_ixs] = False

        val_data[k]['towers'] = val_data[k]['towers'][~train_mask,...]
        val_data[k]['labels'] = val_data[k]['labels'][~train_mask,...]
        
        data[k]['towers'] = data[k]['towers'][train_mask,...]
        data[k]['labels'] = data[k]['labels'][train_mask,...]
        start = end
    return data, val_data


def active_train(ensemble, dataset, val_dataset, dataloader, val_dataloader, data_sampler_fn, data_label_fn, data_pred_fn, data_subset_fn, logger, agent, args):
    """ Main training function 
    :param ensemble: learning.models.Ensemble object to be trained.
    :param dataset: Object containing the data to iterate over. Can be added to.
    :param val_dataset: 
    :param dataloader: The dataloader linked to the given dataset.
    :param val_dataloader:
    :param data_sampler_fn:
    :param data_label_fn:
    :param data_pred_fn:
    :param data_subset_fn:
    :param logger: Object used to keep track of training artifacts.
    :param agent: PandaAgent or None (if args.exec_mode == 'simple-model' or 'noisy-model')
    :param args: Commandline arguments such as the number of acquisition points.
    :return: The fully trained ensemble.
    """
    for tx in range(args.max_acquisitions):
        print('Acquisition Step: ', tx)
        start_time = time.time()
        logger.save_dataset(dataset, tx)

        # Initialize and train models.
        ensemble.reset()
        for model in ensemble.models:
            train(dataloader, val_dataloader, model, args.n_epochs)
        
        logger.save_ensemble(ensemble, tx)

        # Collect new samples.
        new_data, all_samples = acquire_datapoints(ensemble=ensemble, 
                                                   n_samples=args.n_samples, 
                                                   n_acquire=args.n_acquire, 
                                                   strategy=args.strategy,
                                                   data_sampler_fn=data_sampler_fn,
                                                   data_subset_fn=data_subset_fn,
                                                   data_label_fn=data_label_fn,
                                                   data_pred_fn=data_pred_fn,
                                                   exec_mode=args.exec_mode,
                                                   agent=agent)
        logger.save_acquisition_data(new_data, None, tx)#new_data, all_samples, tx)

        # Add to dataset.
        if val_dataloader is None:
            dataset.add_to_dataset(new_data)
        else:
            train_data, val_data = split_data(new_data, n_val=2)
            dataset.add_to_dataset(train_data)
            val_dataset.add_to_dataset(val_data)
            
        print('Time: ' + str((time.time()-start_time)*(1/60)) + ' minutes')
