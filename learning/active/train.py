import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from learning.active.mlp import MLP
from learning.active.toy_data import ToyDataset, ToyDataGenerator
from learning.active.utils import ExperimentLogger


def evaluate(loader, model):
    acc = []
    for x, y in loader:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        # TODO: When doing validation on dropout, average models.
        model.sample_dropout_masks()
        pred = model.forward(x).squeeze()

        accuracy = ((pred>0.5) == y).float().mean()
        acc.append(accuracy.item())

    return np.mean(acc)


def train(dataloader, val_dataloader, model, n_epochs=20):
    optimizer = Adam(model.parameters(), lr=1e-3)
    if torch.cuda.is_available():
        model.cuda()

    for ex in range(n_epochs):
        print('Epoch', ex)
        acc = []
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()

            #model.sample_dropout_masks()
            pred = model.forward(x).squeeze()
            loss = F.binary_cross_entropy(pred, y)
            loss.backward()

            optimizer.step()

            accuracy = ((pred>0.5) == y).float().mean()
            acc.append(accuracy.item())

        print(np.mean(acc))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--n-train', type=int, default=1000)
    parser.add_argument('--n-val', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-models', type=int, default=5, help='Number of models in the ensemble.')
    parser.add_argument('--n-hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--n-epochs', type=int, default=300)
    args = parser.parse_args()

    logger = ExperimentLogger.setup_experiment_directory(args)

    gen = ToyDataGenerator()
    xs, ys = gen.generate_uniform_dataset(N=args.n_train)
    gen.plot_dataset(xs, ys, logger.get_figure_path('dataset_train.png'))
    dataset = ToyDataset(xs, ys)
    logger.save_dataset(dataset, 'train.pkl')

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True)

    val_xs, val_ys = gen.generate_uniform_dataset(N=args.n_val)
    val_dataset = ToyDataset(val_xs, val_ys)
    gen.plot_dataset(val_xs, val_ys, logger.get_figure_path('dataset_val.png'))
    logger.save_dataset(val_dataset, 'val.pkl')
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)

    for mx in range(args.n_models):
        net = MLP(n_hidden=args.n_hidden, dropout=args.dropout)
        best_net = train(loader, val_loader, net, n_epochs=args.n_epochs)

        logger.save_model(best_net, 'net_%d.pt' % mx)
