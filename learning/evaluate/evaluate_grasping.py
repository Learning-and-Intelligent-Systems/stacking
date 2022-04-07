import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import recall_score, precision_score, accuracy_score
from torch.utils.data import DataLoader

from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.grasp_data import GraspDataset, visualize_grasp_dataset


def get_labels_predictions(logger, val_dataset_fname):
    # Load dataset
    with open(val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    val_dataset = GraspDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = logger.get_ensemble(0)
    if torch.cuda.is_available:
        model = model.cuda()
    model.eval()

    # Get predictions
    predictions, labels = [], []
    for x, y in val_dataloader:
        if torch.cuda.is_available():
            x = x.float().cuda()
        with torch.no_grad():
            probs = model.forward(x).mean(dim=1).cpu()

        preds = (probs > 0.5).float()
        
        predictions.append(preds)
        labels.append(y)

    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()

    return labels, predictions

def get_validation_metrics(logger, val_dataset_fname):
    labels, predictions = get_labels_predictions(logger, val_dataset_fname)

    # Calculate metrics
    metrics_fn = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score
    } 

    metrics_val = {}
    for name, fn in metrics_fn.items():
        metrics_val[name] = fn(labels, predictions)
    
    with open(logger.get_figure_path('metrics.json'), 'w') as handle:
        json.dump(metrics_val, handle)
    print(metrics_val)

def visualize_predictions(logger, val_dataset_fname):
    labels, predictions = get_labels_predictions(logger, val_dataset_fname)
    with open(val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    val_dataset = GraspDataset(val_data)


    figure_path = logger.get_figure_path('correct_%d.png')
    visualize_grasp_dataset(val_dataset, 'YcbHammer', 5, 25, labels=predictions==labels, figure_path=figure_path)
    figure_path = logger.get_figure_path('labels_%d.png')
    visualize_grasp_dataset(val_dataset, 'YcbHammer', 5, 25, labels=labels, figure_path=figure_path)
    figure_path = logger.get_figure_path('predictions_%d.png')
    visualize_grasp_dataset(val_dataset, 'YcbHammer', 5, 25, labels=predictions, figure_path=figure_path)

def combine_image_grids(logger, prefixes):
    for ix in range(0, 50):
        for angle in ['x', 'y', 'z']:
            images = []
            for p in prefixes:
                fname = logger.get_figure_path('%s_%d_%s.png' % (p, ix, angle))
                images.append(plt.imread(fname))

            fig = plt.figure(figsize=(5,15))
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 3))

            for ax, im in zip(grid, images):
                print(im.shape)
                im = im[:, 500:, :]
                im = im[:, :-500, :]
                ax.imshow(im)

            for ax, title in zip(grid, prefixes):
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.set_title(title)
            
            plt.savefig(logger.get_figure_path('combined_%d_%s.png' % (ix, angle)), bbox_inches='tight', dpi=500)



            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    parser.add_argument('--val-dataset-fname', type=str, required=True)
    args = parser.parse_args()

    logger = ActiveExperimentLogger(args.exp_path, use_latents=False)

    get_validation_metrics(logger, args.val_dataset_fname)
    #visualize_predictions(logger, args.val_dataset_fname)
    #combine_image_grids(logger, ['labels', 'predictions', 'correct'])