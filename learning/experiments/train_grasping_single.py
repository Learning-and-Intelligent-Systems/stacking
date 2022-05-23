import argparse
import copy
import numpy as np
import pickle
import torch

from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.grasp_data import GraspDataset, GraspParallelDataLoader
from learning.domains.grasping.train_latent import train as train_latent
from learning.models.ensemble import Ensemble
from learning.models.latent_ensemble import GraspingLatentEnsemble
from learning.models.pointnet import PointNetClassifier
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam


def load_datasets(args):
    with open(args.train_dataset_fname, 'rb') as handle:
        train_data = pickle.load(handle)
    with open(args.val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)

    train_dataset = GraspDataset(train_data, grasp_encoding='per_point')
    val_dataset = GraspDataset(val_data, grasp_encoding='per_point')

    if args.use_latents:
        train_dataloader = GraspParallelDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, n_dataloaders=args.n_models)
        val_dataloader = GraspParallelDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, n_dataloaders=1)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataset, val_dataset, train_dataloader, val_dataloader


def initialize_model(args):
    
    if args.model == 'pn':
        base_model = PointNetClassifier
        # base_args = {'n_in': 3+3+3+1+1} # xyz, grasp_indicators, com, mass, friction
        base_args = {'n_in': 3+9+3+1+1-2} # xyz, nxyz, grasp_pts, props
    else:
        raise NotImplementedError()

    ensemble = Ensemble(base_model=base_model,
                        base_args=base_args,
                        n_models=args.n_models)
    if args.use_latents:
        ensemble = GraspingLatentEnsemble(ensemble=ensemble, 
                                          n_latents=args.n_objects, 
                                          d_latents=3)

    return ensemble

def evaluate(loader, model, val_metric='acc'):
    acc = []
    losses = []
    
    preds = []
    labels = []
    model.eval()
    for x, _, y in loader:
        if torch.cuda.is_available():
            x = x.float().cuda()
            y = y.float().cuda()
        pred = model.forward(x).squeeze()
        if len(pred.shape) == 0: pred = pred.unsqueeze(-1)
        loss = F.binary_cross_entropy(pred, y)
     
        with torch.no_grad():
            preds += (pred > 0.5).cpu().float().numpy().tolist()
            labels += y.cpu().numpy().tolist()
        accuracy = ((pred>0.5) == y).float().mean()
        acc.append(accuracy.item())
        losses.append(loss.item())
    if val_metric == 'loss':
        score = np.mean(losses)
    elif val_metric == 'acc':
        score = -np.mean(acc)
    else:
        score = -f1_score(labels, preds)

    return score

def train(dataloader, val_dataloader, model, n_epochs=20):
    """
    :param val_dataloader: If a validation set is given, will return the model
    with the lowest validation loss.
    """
    optimizer = Adam(model.parameters(), lr=1e-3)
    if torch.cuda.is_available():
        model.cuda()

    best_loss = 1000
    best_weights = None
    for ex in range(n_epochs):
        print('Epoch', ex)
        acc = []
        model.train()
        for x, _, y in dataloader:
            if torch.cuda.is_available():
                x = x.float().cuda()
                y = y.float().cuda()
            optimizer.zero_grad()
            
            pred = model.forward(x).squeeze()
            loss = F.binary_cross_entropy(pred, y.squeeze())
            loss.backward()

            optimizer.step()

            accuracy = ((pred>0.5) == y).float().mean()
            acc.append(accuracy.item())
        print('Train Accuracy:', np.mean(acc))
        if val_dataloader is not None:
            val_loss = evaluate(val_dataloader, model, val_metric='acc')
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                print('Saved')    
                print('Val Accuracy:', -val_loss)


    if val_dataloader is not None:
        model.load_state_dict(best_weights)

    return model

def run(args):
    args.use_latents = args.property_repr == 'latent'
    args.fit = False
    print(args)
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # Build model.
    ensemble = initialize_model(args)
    if torch.cuda.is_available():
        ensemble = ensemble.cuda()

    # Train model.
    train_dataset, val_dataset, train_dataloader, val_dataloader = load_datasets(args)
    if args.use_latents:
        train_latent(train_dataloader, val_dataloader, ensemble, args.n_epochs, disable_latents=False, args=args, show_epochs=True)
    else:
        ensemble.reset()
        for model in ensemble.models:
            train(train_dataloader, val_dataloader, model, args.n_epochs)


    # Save model.
    logger.save_dataset(dataset=train_dataset, tx=0)
    logger.save_val_dataset(val_dataset=val_dataset, tx=0)
    logger.save_ensemble(ensemble=ensemble, tx=0)

    return logger.exp_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')

    # Dataset parameters. 
    parser.add_argument('--train-dataset-fname', type=str, required=True)
    parser.add_argument('--val-dataset-fname', type=str, required=True)
    parser.add_argument('--n-objects', type=int, required=True)
    # Model parameters.
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-hidden', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--model', default='pn', choices=['pn', 'pn++'])
    parser.add_argument('--property-repr', type=str, required=True, choices=['latent', 'explicit', 'removed'])
    # Ensemble parameters.
    parser.add_argument('--n-models', type=int, default=1)
    args = parser.parse_args()
    
    run(args)

