import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from learning.active.toy_data import ToyDataset, ToyDataGenerator
from learning.active.mlp import MLP


def evaluate(loader, model):
    acc = []
    for x, y in loader:
        # TODO: When doing validation on dropout, average models.
        model.sample_dropout_masks()
        pred = model.forward(x).squeeze()

        accuracy = ((pred>0.5) == y).float().mean()
        acc.append(accuracy)

    return np.mean(acc)


def train(dataloader, val_dataloader, model, n_epochs=20):
    optimizer = Adam(model.parameters(), lr=1e-3)
    best_acc = 0.
    best_model = None
    for ex in range(n_epochs):
        print('Epoch', ex)
        acc = []
        for x, y in loader:
            optimizer.zero_grad()

            model.sample_dropout_masks()
            pred = model.forward(x).squeeze()
            loss = F.binary_cross_entropy(pred, y)
            loss.backward()

            optimizer.step()

            accuracy = ((pred>0.5) == y).float().mean()
            acc.append(accuracy)

        val_acc = evaluate(val_dataloader, model)
        if val_acc > best_acc:
            print('Saved')
            best_model = MLP(model.n_hidden, model.dropout)
            best_model.load_state_dict(model.state_dict())
            best_acc = val_acc

        print(np.mean(acc))
    return model

def display_marginal_predictions(all_preds, resolution):
    eps = 1e-5
    all_preds = torch.stack(all_preds)

    p = torch.mean(all_preds, dim=0)

    x1 = torch.arange(-1, 1, resolution)
    x2 = torch.arange(-1, 1, resolution)
    x1s, x2s = torch.meshgrid(x1, x2)
    K = x1s.shape[0]
    p = p.view(K, K)
    plt.close()
    plt.pcolormesh(x1s.numpy(), x2s.numpy(), p.numpy())
    plt.savefig('learning/active/figures/preds_ensemble_val_1000_test.png')

def display_bald_objective(all_preds, resolution):
    """
    :param all_preds: A list of predictions for each model.
    :return: A tensor of the BALD value for each predicted point.
    """
    print(all_preds[0].shape)
    eps = 1e-5
    all_preds = torch.stack(all_preds)

    mp_c1 = torch.mean(all_preds, dim=0)
    mp_c0 = torch.mean(1 - all_preds, dim=0)

    m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))

    p_c1 = all_preds
    p_c0 = 1 - all_preds
    ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
    ent = torch.mean(ent_per_model, dim=0)

    bald = m_ent + ent

    x1 = torch.arange(-1, 1, resolution)
    x2 = torch.arange(-1, 1, resolution)
    x1s, x2s = torch.meshgrid(x1, x2)
    K = x1s.shape[0]
    bald = bald.view(K, K)
    plt.close()
    plt.pcolormesh(x1s.numpy(), x2s.numpy(), bald.numpy())
    plt.savefig('learning/active/figures/bald_ensemble_val_1000_test.png')

res=0.005
exp_name = 'ensemble'
if __name__ == '__main__':
    gen = ToyDataGenerator()
    xs, ys = gen.generate_uniform_dataset(N=1000)
    gen.plot_dataset(xs, ys, 'learning/active/figures/dataset_1000.png')
    dataset = ToyDataset(xs, ys)
    loader = DataLoader(dataset,
                        batch_size=16,
                        shuffle=True)

    val_xs, val_ys = gen.generate_uniform_dataset(N=500)
    val_dataset = ToyDataset(val_xs, val_ys)
    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False)

    all_preds = []
    for mx in range(50):
        net = MLP(n_hidden=128, dropout=0.)

        best_net = train(loader, val_loader, net, n_epochs=300)

        # Returns a list of the predictions for each of the dropout models.
        all_preds += best_net.plot_decision_boundary(resolution=res, fname='tmp%d' % mx, k=1)

    # TODO: Plot the BALD objective values for each of the contour points.
    display_bald_objective(all_preds, resolution=res)
    display_marginal_predictions(all_preds, resolution=res)