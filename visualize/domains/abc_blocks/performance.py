import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def detailed_error_stats(args, trans_dataset, trans_model):
    def tensor_to_tuple(edge_tensor):
        return tuple(map(tuple, edge_tensor.squeeze().numpy()))
    # get all unique transitions in dataset and collect stats
    xs, ys = trans_dataset[:]
    all_preds = trans_model(xs).detach()
    trans_info = {}
    for i, ((object_features, edge_features, action), next_edge_features) in enumerate(trans_dataset):
        trans_i = tensor_to_tuple(edge_features)
        action = tuple(action.numpy())
        trans_j = tensor_to_tuple(next_edge_features)
        trans = (trans_i, action, trans_j)
        if trans in trans_info:
            trans_info[trans]['frequency'] += 1
            trans_info[trans]['preds'] += [all_preds[i]]
            trans_info[trans]['true_preds'] = [next_edge_features.squeeze()]
        else:
            trans_info[trans] = {}
            trans_info[trans]['frequency'] = 1
            trans_info[trans]['preds'] = [all_preds[i]]
            trans_info[trans]['true_preds'] = [next_edge_features.squeeze()]

    n_ef = 1 # number of edge features
    N_OBJECTS = len(object_features) # hack
    # Print results
    for trans in trans_info:
        
        preds_expand = [pred.view(1, N_OBJECTS, N_OBJECTS, n_ef) for pred in trans_info[trans]['preds']]
        preds_tensor = torch.cat(preds_expand, axis=0)
        round_preds_tensor = preds_tensor.round()
        avg_preds_tensor = round_preds_tensor.mean(axis=0).squeeze() + 0 # change -0 to 0
        
        edge_features, action, next_edge_features = trans
        mse = np.linalg.norm(avg_preds_tensor - np.array(next_edge_features))
        
        print('Frequency %f' % trans_info[trans]['frequency'])
        print('Initial state')
        print_edge_state(edge_features)
        print('Action')
        print_action(action)
        if mse > 0:
            print('Predicted delta state')
            print_edge_state(tensor_to_tuple(avg_preds_tensor))
        print('------------------')

def print_edge_state(edge_state):
    for row in edge_state:
        print(row)
        
def print_action(action):
    bottom_block = action[0]
    top_block = action[1]
    #bottom_block = int(np.where(np.array(action[MAX_OBJECTS:]) == 1.)[0])
    #top_block = int(np.where(np.array(action[:MAX_OBJECTS]) == 1.)[0]) 
    print('%i --> %i' % (top_block, bottom_block))
    
def print_trans(trans_tuple):
    print('initial_edge_state')
    print_edge_state(trans_tuple[0])
    print('action')
    print_action(trans_tuple[1])
    print('next_edge_state')
    print_edge_state(trans_tuple[2])
    
    
def calc_heur_error_rate(heur_dataset, model):
    n = len(heur_dataset)
    xs, ys = heur_dataset[:]
    preds = model(xs)
    
    # on average how many edges are predicted incorrectly per datapoint
    error_rate = torch.sqrt((preds-ys)**2).mean()
    return error_rate    
    
def vis_trans_errors(args, test_dataset, model):
    fig, ax = plt.subplots()
    xs, ys = test_dataset[:]
    preds = model(xs)
    # N number of datapoints
    # K objects
    # E edge types
    N, K, K, E = ys.shape 
    
    for ei in range(E):
        preds_ei = preds[:,:,:,ei].round()
        ys_ei = ys[:,:,:,ei]
        errors = (preds_ei != ys_ei).float().sum(axis=0).detach()
        avg_errors = np.nan_to_num(errors/N)
        
        c = ax.pcolor(avg_errors, cmap='viridis', vmin=0, vmax=1)
        ax.set_title('Erroneous Edge Predictions in Test Dataset\nEdge Type %i' % ei)
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
    # # HACK: 
    TABLE = 0

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
    