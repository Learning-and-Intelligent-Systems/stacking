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
    
def calc_successful_action_error_rate(trans_dataset, trans_model):
    xs, ys = trans_dataset[:]
    preds = trans_model(xs)
    ## for full next state prediction
    #action_errors = []
    #for i, ((state, action), next_state) in enumerate(trans_dataset):
        #if (state != next_state).any(): 
            #action_errors.append(((preds[i]>0.5) != next_state).sum().float())
    ## for delta_state prediction
    actions, action_errors = 0, 0
    for i, ((state, action), delta_state) in enumerate(trans_dataset):
        num_edge_changes = delta_state.abs().sum()
        if num_edge_changes != 0:
            actions += num_edge_changes
            action_errors += (preds[i].round() != delta_state).sum().float()
            print(preds[i])
    print('%i/%i datapoints consisted of transtitions.' % (actions, len(trans_dataset)))
    print('%i/%i transitions were incorrectly predicted' % (action_errors, actions))
    
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
    
def plot_error_stats(n_datapoints, error_stats):
    fig, ax = plt.subplots()
    
    avg_error = np.array(error_stats).mean(axis=0)
    std_error = np.array(error_stats).std(axis=0)
    print('Average Error over %i runs: %f' % (len(error_stats), avg_error))
    print('St Dev Error over %i runs: %f' % (len(error_stats), std_error))
    ax.plot(n_datapoints, avg_error)
    ax.fill_between(n_datapoints, avg_error-std_error, avg_error+std_error, alpha=0.2)
    