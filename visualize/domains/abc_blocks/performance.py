import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def calc_trans_error_rate(args, trans_dataset, model):
    n = len(trans_dataset)
    xs, ys = trans_dataset[:]
    preds = model(xs)
    
    # on average how many edges are predicted incorrectly per datapoint
    if args.pred_type == 'full_state':
        # wrong predictions out of all possible predictions
        error_rate = ((preds>0.5) != ys).sum(dim=(1,2)).float().mean()
    elif args.pred_type == 'delta_state':
        # wrong predictions about of possible predictions
        # TODO: this is not correct
        error_rate = 0
        #error_rate = (preds[i].round() != delta_state).sum().float().mean()
    return error_rate
    
def calc_successful_action_error_rate(args, trans_dataset, trans_model):
    xs, ys = trans_dataset[:]
    preds = trans_model(xs)
    if args.pred_type == 'full_state':
        action_errors = 0
        successful_actions = 0
        for i, ((object_features, edge_features, action), next_edge_features) in enumerate(trans_dataset):
            if (edge_features != next_edge_features).any(): 
                successful_actions += 1
                if ((preds[i]>0.5) != next_edge_features).sum().float() > 0.:
                    action_errors += 1
                print(preds[i,:,:,0], next_edge_features[:,:,0])
                #action_errors.append(((preds[i]>0.5) != next_state).sum().float())
        print('%i/%i datapoints consisted of transtitions.' % (successful_actions, len(trans_dataset)))
        print('%i/%i transitions were incorrectly predicted' % (action_errors, successful_actions))
    elif args.pred_type == 'delta_state':
        actions, action_errors = 0, 0
        for i, ((object_features, edge_features, action), delta_edge_features) in enumerate(trans_dataset):
            num_edge_changes = delta_edge_features.abs().sum()
            if num_edge_changes != 0:
                actions += 1
                if (preds[i].round() != delta_edge_features).sum().float() > 0.:
                    action_errors += 1
                print(preds[i,:,:,0])
                #action_errors += (preds[i].round() != delta_state).sum().float()
                #print(preds[i])
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
    