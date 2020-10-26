import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from matplotlib.animation import FuncAnimation, writers
from torch.utils.data import DataLoader

from learning.domains.toy2d.active_utils import get_predictions
from learning.domains.toy2d.toy_data import ToyDataset
from learning.active.utils import ActiveExperimentLogger


def get_ensemble_predictions(logger, resolution=0.02, eps=1e-5):
    # Load a gridded dataset. 
    x1 = torch.arange(-1, 1, resolution)
    x2 = torch.arange(-1, 1, resolution)

    x1s, x2s = torch.meshgrid(x1, x2)
    K = x1s.shape[0]
    x = torch.cat([x1s.reshape(K*K, 1), x2s.reshape(K*K, 1)], dim=1)
    
    dataset = ToyDataset(x, torch.zeros((K*K,)))
    preds = []
    balds = []

    # For each tx, find the predictions for each point in the grid.
    for tx in range(logger.args.max_acquisitions):
        print(tx)
        ensemble = logger.get_ensemble(tx)
        pred = get_predictions(dataset, ensemble)
        preds.append(pred.mean(dim=1).view(K, K))
        mp_c1 = torch.mean(pred, dim=1)
        mp_c0 = torch.mean(1 - pred, dim=1)
        m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))
        p_c1 = pred
        p_c0 = 1 - pred
        ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
        ent = torch.mean(ent_per_model, dim=1)
        bald = m_ent + ent
        balds.append(bald.view(K, K))

    return (x1s, x2s, preds, balds)

def get_datasets_and_acquisitions(logger):
    data = []
    acquired_points = []
    n_active = logger.args.max_acquisitions
    for tx in range(n_active):
        data.append(logger.load_dataset(tx))
        acquired_points.append(logger.load_acquisition_data(tx)[0])
    return data, acquired_points

def create_animation(logger, data, acquired_points, preds_info):
    # Animate the list of predictions.
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    x1s, x2s, preds, balds = preds_info

    def init():
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])

        # Draw predictions
        contour = ax2.pcolormesh(x1s.numpy(), x2s.numpy(), preds[0].numpy())
        ax3.pcolormesh(x1s.numpy(), x2s.numpy(), balds[0].numpy())
        
        # Plot the data.
        ax1.scatter(data[0].xs[:, 0], data[0].xs[:, 1], s=4, c=data[0].ys)
        ax1.scatter(acquired_points[0][:,0], acquired_points[0][:,1])

        return contour,

    def update(frame_ix):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.set_title('Data (T=%d)' % frame_ix)
        ax2.set_title('Predictions (T=%d)' % frame_ix)
        ax3.set_title('BALD (T=%d)' % frame_ix)
        
        p = preds[frame_ix]
        b = balds[frame_ix]
        print(p.shape, b.shape)
        contour = ax2.pcolormesh(x1s.numpy(), x2s.numpy(), p.numpy())
        ax3.pcolormesh(x1s.numpy(), x2s.numpy(), b.numpy())

        ax1.scatter(acquired_points[frame_ix][:,0], acquired_points[frame_ix][:,1])
        ax1.scatter(data[frame_ix].xs[:, 0], data[frame_ix].xs[:, 1], s=4, c=data[frame_ix].ys)
        return contour,

    Writer = writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    ani = FuncAnimation(fig, update, frames=range(len(data)), interval=500)
    path = logger.get_figure_path('active.mp4')
    ani.save(path, writer=writer)
    
    # plt.show()

def get_validation_accuracy(logger, val_path):
    # Load the validation dataset.
    with open(val_path, 'rb') as handle:
        val_dataset = pickle.load(handle)

    accs = []
    
    for tx in range(logger.args.max_acquisitions):
        # Load the ensemble and validate the predictions.
        ensemble = logger.get_ensemble(tx)
        preds = get_predictions(val_dataset, ensemble).numpy().mean(1)
        print(preds.shape, val_dataset.ys.shape)
        acc = (preds > 0.5) == val_dataset.ys
        accs.append(acc.mean())
        print(accs[-1])

    log_path = logger.get_figure_path('val_accs.pkl')
    with open(log_path, 'wb') as handle:
        pickle.dump(accs, handle)


def make_validation_curves():
    with open('learning/active/experiments/exp-20201019-211808/figures/val_accs.pkl', 'rb') as handle:
        bald_accs = pickle.load(handle)
    with open('learning/active/experiments/random-20201020-103928/figures/val_accs.pkl', 'rb') as handle:
        random_accs = pickle.load(handle)

    xs = np.arange(100, 2100, 10)
    bald_xs = xs[:len(bald_accs)]
    random_xs = xs[:len(random_accs)]

    plt.plot(bald_xs, bald_accs, label='bald')
    plt.plot(random_xs, random_accs, label='random')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.savefig('val_curve.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()

    logger = ActiveExperimentLogger(args.exp_path)
    logger.args.max_acquisitions = 10
    #make_validation_curves()
    get_validation_accuracy(logger, 'learning/evaluate/val_dataset.pkl')

    data, acquired_points = get_datasets_and_acquisitions(logger)
    pred_info = get_ensemble_predictions(logger)
    create_animation(logger, data, acquired_points, pred_info)