import argparse

from torch.utils.data import DataLoader

from learning.active.active_train import active_train
from learning.domains.toy2d.active_utils import get_labels, get_predictions, sample_unlabeled_data, get_subset
from learning.domains.toy2d.toy_data import ToyDataset, ToyDataGenerator
from learning.models.ensemble import Ensemble
from learning.models.mlp_dropout import MLP
from learning.active.utils import ActiveExperimentLogger


def run_active_toy2d(args):
    logger = ActiveExperimentLogger.setup_experiment_directory(args)
    
    # Initialize ensemble.
    ensemble = Ensemble(base_model=MLP,
                        base_args={'n_hidden': args.n_hidden, 'dropout': args.dropout},
                        n_models=args.n_models)

    # Sample initial dataset.
    gen = ToyDataGenerator()
    xs, ys = gen.generate_uniform_dataset(N=args.n_train_init)
    dataset = ToyDataset(xs, ys)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True) 

    active_train(ensemble=ensemble, 
                 dataset=dataset, 
                 dataloader=dataloader, 
                 data_sampler_fn=sample_unlabeled_data, 
                 data_label_fn=get_labels, 
                 data_pred_fn=get_predictions,
                 data_subset_fn=get_subset,
                 logger=logger, 
                 args=args)


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
    parser.add_argument('--n-epochs', type=int, default=500)
    parser.add_argument('--n-train-init', type=int, default=100)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--n-acquire', type=int, default=10)
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--strategy', choices=['random', 'bald'], default='bald')    
    args = parser.parse_args()

    run_active_toy2d(args)