import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from learning.domains.abc_blocks.world import TABLE, MAXBLOCK


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
    
def detailed_error_stats(args, trans_dataset, trans_model):
    n = len(trans_dataset)
    trans_types = 5
    transition_frequency = np.zeros(trans_types)
    preds = [[] for _ in range(trans_types)]
    xs, ys = trans_dataset[:]
    all_preds = trans_model(xs).detach()
    corr_pred = np.zeros((trans_types,3,3))
    for i, ((object_features, edge_features, action), next_edge_features) in enumerate(trans_dataset):
        if edge_features[TABLE, 1] == 1 and edge_features[TABLE, 2] ==1:
            # Transition 1: all blocks on table, then place(A)
            if (action == torch.tensor([0, 1, 0])).all():
                print('Transition 1')
                print(edge_features.squeeze())
                print(next_edge_features.squeeze())
                transition_frequency[0] += 1
                preds[0] += [all_preds[i]]
                corr_pred[0] = next_edge_features.squeeze()
            # Transition 2: all blocks on table, then place(B)
            elif (action == torch.tensor([0, 0, 1])).all():
                transition_frequency[1] += 1
                preds[1] += [all_preds[i]]
                corr_pred[1] = next_edge_features.squeeze()
                print('Transition 2')
                print(edge_features.squeeze())
                print(next_edge_features.squeeze())
        # A on B
        elif edge_features[TABLE, 1] == 1 and edge_features[1, 2] ==1:
            if (action == torch.tensor([0, 1, 0])).all():
                transition_frequency[2] += 1
                preds[2] += [all_preds[i]]
                corr_pred[2] = next_edge_features.squeeze()
            elif (action == torch.tensor([0, 0, 1])).all():
                transition_frequency[3] += 1
                preds[3] += [all_preds[i]]
                corr_pred[3] = next_edge_features.squeeze()
        # Non action
        elif (action == torch.tensor([0, 0, 0])).all():
            transition_frequency[4] += 1
            preds[4] += [all_preds[i]]
            corr_pred[4] = next_edge_features.squeeze()
    for type in range(trans_types):
        print('Transition type %i occured %f of the time in the training datasest' % (type, transition_frequency[type]/n))
        if transition_frequency[type] != 0:
            print('Its average prediction was')
            print(torch.cat(preds[type], axis=2).mean(axis=2))
            print('Correct Prediction')
            print(corr_pred[type])
                
                
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
                print(preds[i].squeeze(), next_edge_features.squeeze())
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
                print(preds[i].squeeze())
                print(delta_edge_features.squeeze())
                #action_errors += (preds[i].round() != delta_state).sum().float()
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
    