import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def calc_trans_error_rate(trans_dataset, model):
    n = len(trans_dataset)
    xs, ys = trans_dataset[:]
    preds = model(xs)
    
    # on average how many edges are predicted incorrectly per datapoint
    error_rate = ((preds>0.5) != ys).sum(dim=(1,2)).float().mean()
    return error_rate
    
def calc_heur_error_rate(heur_dataset, model):
    n = len(heur_dataset)
    xs, ys = heur_dataset[:]
    preds = model(xs)
    
    # on average how many edges are predicted incorrectly per datapoint
    error_rate = torch.sqrt((preds-ys)**2).mean()
    return error_rate    
    
def vis_trans_errors(test_dataset, model):
    fig, ax = plt.subplots()
    xs, ys = test_dataset[:]
    preds = model(xs)
    
    edges_in_dataset = ys.sum(axis=0).detach()
    errors = ((preds>0.5) != ys).float().sum(axis=0).detach()
    
    avg_errors = np.nan_to_num(errors/edges_in_dataset)
    c = ax.pcolor(avg_errors, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Erroneous Edge Predictions in Test Dataset')
    ax.set_ylabel('Bottom Object')
    ax.set_xlabel('Top Object')
    fig.colorbar(c)
    
    
def vis_trans_dataset_grid(trans_dataset, title):
    fig, ax = plt.subplots()
    xs, ys = trans_dataset[:]
    all_edge_masks, all_actions = xs
    avg_edge_mask = all_edge_masks.sum(axis=0).detach()
    c = ax.pcolor(avg_edge_mask, cmap='viridis')
    ax.set_title(title)
    ax.set_ylabel('Bottom Object')
    ax.set_xlabel('Top Object')
    fig.colorbar(c)
    
def vis_trans_dataset_hist(trans_dataset, title):
    # Visualize Training Dataset
    stacked_blocks = []
    for x, y in trans_dataset:
        state, action = x 
        state_stacked = state[2:,2:].sum()+1  # one block is stacked on table
        stacked_blocks.append(state_stacked)
    
    fig, ax = plt.subplots()
    ax.hist(stacked_blocks)
    ax.set_title(title)
    ax.set_xlabel('Number of Blocks in Stack')
    ax.set_ylabel('Frequency')