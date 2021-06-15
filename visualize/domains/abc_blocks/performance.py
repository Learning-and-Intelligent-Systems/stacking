import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from learning.domains.abc_blocks.world import TABLE, MAXBLOCK, get_obj_one_hot, N_OBJECTS

def detailed_error_stats(args, trans_dataset, trans_model):
    # get all unique transitions in dataset
    trans_info = {}
    for i, ((object_features, edge_features, action), next_edge_features) in enumerate(trans_dataset):
        trans_i = tuple(map(tuple, edge_features.squeeze().numpy()))
        trans_j = tuple(map(tuple, next_edge_features.squeeze().numpy()))
        trans = (trans_i, trans_j)
        if trans not in trans_info:
            trans_info[trans] = {}
            trans_info[trans]['frequency'] = 0
            trans_info[trans]['preds'] = []
            trans_info[trans]['true_preds'] = []

    # get stats on each transition type
    xs, ys = trans_dataset[:]
    all_preds = trans_model(xs).detach()
    for i, ((object_features, edge_features, action), next_edge_features) in enumerate(trans_dataset):
        trans_i = tuple(map(tuple, edge_features.squeeze().numpy()))
        trans_j = tuple(map(tuple, next_edge_features.squeeze().numpy()))
        trans_key = (trans_i, trans_j)
        trans_info[trans_key]['frequency'] += 1
        trans_info[trans_key]['preds'] += [all_preds[i]]
        trans_info[trans_key]['true_pred'] = next_edge_features.squeeze()

    n_ef = 1 # number of edge features
    # Print results
    for trans in trans_info:
        print('Transition type')
        print_trans(trans)
        print('Frequency %f' % trans_info[trans]['frequency'])
        print('Average prediction')
        preds_expand = [pred.view(1, N_OBJECTS, N_OBJECTS, n_ef) for pred in trans_info[trans]['preds']]
        preds_tensor = torch.cat(preds_expand, axis=0)
        avg_preds = preds_tensor.mean(axis=0)
        if n_ef == 1:
            avg_preds = avg_preds.squeeze()
        print(avg_preds)
        print('Correct Prediction')
        print(trans_info[trans]['true_pred'])

def print_trans(trans_tuple):
    for edge_state in trans_tuple:
        print(edge_state[0])
        print(edge_state[1])
        print(edge_state[2])
        print('-----------')

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
    
    
def vis_trans_dataset_grid(args, trans_dataset, title):
    fig, ax = plt.subplots()
    xs, ys = trans_dataset[:]
    if args.pred_type == 'full_state':
        new_edge_features = ys
    elif args.pred_type == 'delta_state':
        _, edge_features, _ = xs
        delta_edge_features = ys
        new_edge_features = torch.add(edge_features, delta_edge_features)
    avg_edge_features = new_edge_features.sum(axis=0).detach().squeeze()
    c = ax.pcolor(avg_edge_features, cmap='viridis')
    ax.set_title(title)
    ax.set_ylabel('Bottom Object')
    ax.set_xlabel('Top Object')
    fig.colorbar(c)
    
def vis_trans_dataset_hist(args, trans_dataset, title):
    # Visualize Training Dataset
    stacked_blocks = []
    for x, y in trans_dataset:
        if args.pred_type == 'full_state':
            new_edge_features = y
        elif args.pred_type == 'delta_state':
            _, edge_features, _ = x
            delta_edge_features = y
            new_edge_features = torch.add(edge_features, delta_edge_features)
        state_stacked = new_edge_features[TABLE+1:,TABLE+1:].sum()+1  # one block is stacked on table
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
    