# Active Learning Infrastructure

## Training Loop
Generally, the active learning will implement the following algorithm:

```
active_train:
    1. Get initial dataset (randomly sample datapoints).
    2. Loop
        2a. Train new ensemble with current dataset.
        2b. Collect new samples according to BALD objective.
            - Sample candidate datapoints.
            - Evaluate each according to the BALD objective.
            - Take the top examples to add to the dataset.
```

## Logging
Throughout training, many artifacts will be produced that can be used for further analysis later on. Our approach is to not do any of this analysis during training, but to save all relevant artifacts to produce the analysis later on. This will allow us to perform analyses without
the need to retrain. 

We should we saving all datasets, models, and acquisition points for post-hoc analysis.

The experiments directory will be used for this and each active learning experiment ill have a folder with the corresponding subdirectory structure:

```
experiments/
--<exp-name>-<timestamp>/
----acquisition-data/ (Each folder will contain the set of all points considered in the sampler. A second file will contain a list of the points that were actually chosen.)
----datasets/ (There will be a dataset for each set of ensembles in this folder. The ids will correspond with the models.)
----models/
------<active-loop-ix>/ (There will be a folder with all the trained models after each acquisition. 0 corresponds to the model with the initial data)
----figures/ (Will store posthoc results, not changed by the active_train script)
```

## Evaluations
The main evaluation we care about is validation performance as a function of number of datapoints. This can easily be computed with stored models and a separate validation set.

Some other interesting/useful analyses:
1) Evolution of where the data points are being sampled from.
2) Evolution of the BALD objective.
3) Evolution of the ensemble predictions.
It would be great to turn these into animated gifs.

## API
def active_train(args):
    """ Main training function 
    :param args: Commandline arguments such as the number of acquisition points.
    :return: Ensembles. The fully trained ensembles.
    """

def train(dataloader, val_dataloader, model, n_epochs):
    """ Function to train a single model. Already implemented in train.py 
    :param dataloader: Dataloader for the current dataset.
    :param mode: Model to be trained.
    :param n_epochs: Number of epochs to train the model for.
    :return: The trained model.
    """

def acquire_datapoints(ensemble, n_samples, n_acquire):
    """ Get new datapoints given the current set of models.
    Calls the next three methods in turn with their respective 
    parameters.
    :return: (n_acquire, 2), (n_acquire,) - x,y tuples of the new datapoints.
    """

def sample_unlabelled_data(n_samples):
    """ Randomly sample datapoints without labels. 
    :param n_samples: The number of samples to return.
    :return: np.array(n_samples, 2)
    """

def choose_acquisition_data(samples, ensemble, n_acquire):
    """ Choose data points with the highest acquisition score
    :param samples: An array of unlabelled datapoints which to evaluate.
    :param ensemble: A list of models. 
    :param n_acquire: The number of data points to acquire.
    :return: (n_acquire, 2) - the samples which to label.
    """

def get_labels(samples):
    """ Get the labels for the chosen datapoints.
    :param samples: (n_acquire, 2)
    :return: (n_acquire,) The labels for the given datapoints.
    """

def add_to_dataset(dataset, new_xs, new_ys):
    """ Create a new dataset by adding the new points to the current data.
    :param dataset: The existing ToyDataset object.
    :param new_xs: (n_acquire, 2) 
    :param new_ys: (n_acquire,)
    :return: A new ToyDataset instance with all datapoints.
    """

def bald(predictions):
    """ Get the BALD score for each example.
    :param predictions: (N, K) predictions for N datapoints from K models.
    :return: (N,) The BALD score for each of the datapoints.
    """