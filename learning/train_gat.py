
""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from matplotlib import pyplot as plt
import numpy as np
import pickle
from random import shuffle
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.analyze_data import is_geometrically_stable, is_com_stable, get_geometric_thresholds
from learning.gat import FCGAT
from learning.gn import FCGN
from learning.mlp import MLP
from learning.lstm import TowerLSTM
from learning.gated_gn import GatedGN
from learning.cnn import TowerCNN
from learning.conv_rnn import TowerConvRNN
from learning.conv_rnn_small import TowerConvRNNSmall

def get_subsets(data):
    towers, labels = [], []
    for ix in range(data['towers'].shape[0]):
        tower = data['towers'][ix,:,:]
        label = data['labels'][ix]
        g_stable = is_geometrically_stable(tower[0,:], tower[1,:])
        c_stable = is_com_stable(tower[0,:], tower[1,:])
        if label != c_stable:
            continue
        if (not g_stable and label) or (g_stable and not label):
            towers.append(tower)
            labels.append(label)
    return torch.Tensor(towers), torch.Tensor(labels)

def load_dataset(name, args):
    """ Load all the tower data into TensorDatasets. We need a different
    dataset for each tower size, because vectorized Graph Attention Network
    can only ingest batches of graphs with equal numbers of nodes.

    Arguments:
        name {string} -- dataset name

    Returns:
        list(TensorDataset) -- datasets for each tower size
    """
    with open(f'learning/data/{name}', 'rb') as f:
        all_data = pickle.load(f)
    #all_num_blocks = [int(num_blocks[0]) for num_blocks in all_data.keys()]
    all_num_blocks = [2, 3]
    datasets = []
    for num_blocks in all_num_blocks:
        data = all_data[f'{num_blocks}block']
        # load the tower data
        towers = torch.Tensor(data['towers'])
        labels = torch.Tensor(data['labels'])
        if args.visual:
            images = torch.Tensor(data['images'])

        #towers, labels = get_subsets(data)
        # remove the three color channels at the end of each block encoding
        # (see block_utils.Object.vectorize for details)
        towers = towers[...,:14]
        #towers = towers[...,[0, 1, 2, 4, 5, 7, 8]]
        # convert absolute xy positions to relative positions
        #towers[:,1:,7:9] -= towers[:,:-1,7:9]
        # add the new dataset to the list of datasets
        if args.visual:
            datasets.append(TensorDataset(towers, labels, images))
        else:
            datasets.append(TensorDataset(towers, labels))

    return datasets, all_num_blocks

def print_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c-a  # free inside cache
    print('allocated:', a, 'cached:', c)

def print_split_accuracies(dataset, model):
    """
    Subdivide the accuracies into multiple groups.
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    correct_dict = {}
    count_dict = {}
    for g in [0, 1]:
        correct_dict[g] = {}
        count_dict[g] = {}
        for c in [0, 1]:
            correct_dict[g][c] = 0
            count_dict[g][c] = 0

    total = 0
    thresholds = []
    for tower, label in dataloader:
        g_stable = is_geometrically_stable(tower[0,0,:], tower[0,1,:])
        c_stable = int(label)
        #pred = model.iterate(tower, k=1)
        pred = model.forward(tower, k=tower.shape[1]-1)
        correct = ((pred > 0.5) == c_stable).float().mean()
        if correct < 0.5:
            thresholds += get_geometric_thresholds(tower[0,0,:], tower[0,1,:])

        count_dict[g_stable][c_stable] += 1
        correct_dict[g_stable][c_stable] += correct

        total += 1
        if total > 2000: break
    print(count_dict, correct_dict)
    print('Thresholds:', np.mean(thresholds))
    for g in [0, 1]:
        for c in [0, 1]:
            if count_dict[g][c] == 0:
                acc = -1
            else:
                acc = correct_dict[g][c]/count_dict[g][c]
            print('Geom %d, CoM %d: %f' % (g, c, acc))


def train(model, datasets, test_datasets, args):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    num_data_points = len(datasets[0])
    epoch_ids, test_accuracies = [], []

    for epoch_idx in range(args.epochs):
        print('epoch', epoch_idx)
        #print_memory()
        # create a dataloader for each tower size
        iterable_dataloaders = [
            iter(DataLoader(d, batch_size=args.batch_size, shuffle=True))
            for d in datasets]
        for batch_idx in range(num_data_points // args.batch_size):
            #print('batch', batch_idx)
            #print_memory()
            # shuffle(iterable_dataloaders)
            # iterate through the tower sizes in the inner loop
            for dx, iterable_dataloader in enumerate(iterable_dataloaders):
                
                optimizer.zero_grad()
                if args.visual:
                    towers, labels, images = next(iterable_dataloader)
                else:
                    towers, labels = next(iterable_dataloader)
                if torch.cuda.is_available():
                    towers = towers.cuda()
                    labels = labels.cuda()
                    if args.visual:
                        images = images.cuda()
                if args.visual:
                    preds = model.forward(images, k=towers.shape[1]-1)
                else:
                    preds = model.forward(towers, k=towers.shape[1]-1)
                l = F.binary_cross_entropy(preds, labels)
                l.backward()
                optimizer.step()
                accuracy = ((preds.cpu().detach().numpy().squeeze()>0.5) == labels.cpu().detach().numpy()).mean()
                train_losses.append(accuracy.item())
        print('training acc: ', np.mean(train_losses[-(num_data_points // args.batch_size):]))
        # train_losses is a vector of running average accuracies for each tower size

        #print(preds, labels)
            


            #if batch_idx % 40 == 0:
            #    print(f'Epoch {epoch_idx}\tBatch {batch_idx}:\t {train_losses[-4:]}')
                
        if epoch_idx % 5 == 0:
            epoch_ids += [epoch_idx]

            accuracies = test(model, test_datasets, args, fname='lstm_preds.pkl')
            test_accuracies.append(accuracies)
            print('test acc: ', accuracies)
            #print('Val:', accuracies)
        
        
        #print('EPOCH: Total memory, cached, allocated [GiB]:', t/1073741824, c/1073741824, a/1073741824)
        
        #print_split_accuracies(datasets[0], model)
        
    # save model
    torch.save(model.state_dict(), 'model.pt')
    
    return train_losses, epoch_ids, test_accuracies

def test(model, datasets, args, fname=''):
    accuracies = []
    
    #results = []
    # iterate through the tower sizes
    for dataset in datasets:
        if args.visual:
            # have to run test set through network in batches due to memory issues
            num_data_points = len(dataset)
            batch_size = 100
            dataloader = iter(DataLoader(dataset, batch_size=batch_size))
            accs = []
            for batch_idx in range(num_data_points // batch_size):
                #print('testing batch: ', batch_idx)
                #print_memory()
                towers, labels, images = next(dataloader)
                if torch.cuda.is_available():
                    towers = towers.cuda()
                    labels = labels.cuda()
                    images = images.cuda()
                    
                preds = model.forward(images, k=towers.shape[1]-1)
                accuracy = ((preds.cpu().detach().numpy().squeeze()>0.5) == labels.cpu().detach().numpy()).mean()
                accs.append(accuracy.item())
                #results.append((towers.cpu(), labels.cpu(), preds.cpu().detach().numpy()))
            accuracies.append(np.mean(accs))
        else:
            # pull out the input and output tensors for the whole dataset
            towers = dataset[:][0]
            labels = dataset[:][1]
            if torch.cuda.is_available():
                towers = towers.cuda()
                labels = labels.cuda()
                    
            # run the model on everything
            preds = model.forward(towers, k=towers.shape[1]-1)
            # calculate and save the accuracy
            accuracy = ((preds.cpu().detach().numpy()>0.5) == labels.cpu().detach().numpy()).mean()
            accuracies.append(accuracy.item())
            #results.append((towers.cpu(), labels.cpu(), preds.cpu().detach().numpy()))
    #if len(fname) > 0:
    #    with open(fname, 'wb') as handle:
    #        pickle.dump(results, handle)
    return accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NOTE(caris): this has only been tested for the GatedGN network!!
    parser.add_argument('--visual', action='store_true') 
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--n-hidden', default=32, type=int)
    args = parser.parse_args()
    
    if args.debug:
        import pdb; pdb.set_trace()
    
    # needed to work with lis-cloud GPUs
    torch.backends.cudnn.enabled = False
    
    # the number of hidden variables in the graph NN
    M = 64
    #model = FCGAT(14+M, M)
    #model = MLP(5, 256)
    #model = FCGN(14, 64, visual=args.visual, image_dim=150)
    #model = TowerLSTM(14, args.n_hidden, visual=args.visual, image_dim=150)
    #model = TowerCNN(150)
    #model = TowerConvRNN(150)
    model = TowerConvRNNSmall(150)
    #model = GatedGN(14, 32, visual=args.visual, image_dim=150)
    if torch.cuda.is_available():
        model = model.cuda()
    train_dataset = 'random_blocks_(x10000)_2to5blocks_uniform_density.pkl'
    test_dataset = 'random_blocks_(x2000)_2to5blocks_uniform_density.pkl'
    train_datasets, _ = load_dataset(train_dataset, args)
    test_datasets, num_test_blocks = load_dataset(test_dataset, args)
    
    train_losses, epoch_ids, test_accuracies = train(model, train_datasets, test_datasets, args)
    fig, ax = plt.subplots()
    ax.plot(train_losses)
    ax.set_title('Training Loss')
    ax.set_xlabel('Batch (x10)')
    fig.savefig('train_losses.png')
    plt.close()
    
    fig, ax = plt.subplots()
    ax.plot(epoch_ids, test_accuracies, label=num_test_blocks)
    ax.legend(title='number of blocks')
    ax.set_title('Test Accuracy')
    ax.set_xlabel('Epoch ID')
    ax.set_ylabel('Accuracy')
    fig.savefig('test_accuracies.png')
    
    # write training params to a file
    model_type = None
    if isinstance(model, FCGAT):
        model_type = 'FCGAT'
    elif isinstance(model, MLP):
        model_type = 'MLP'
    elif isinstance(model, TowerLSTM):
        model_type = 'TowerLSTM'
    elif isinstance(model, FCGN):
        model_type = 'FCGN'
    elif isinstance(model, GatedGN):
        model_type = 'GatedGN'
    elif isinstance(model, TowerCNN):
        model_type = 'TowerCNN'
    elif isinstance(model, TowerConvRNN):
        model_type = 'TowerConvRNN'
    elif isinstance(model, TowerConvRNNSmall):
        model_type = 'TowerConvRNNSmall'
    else:
        print('Model type could not be determined.')
    
    file = open("params.txt","w")
    file.write("model type : " + model_type + " \n") 
    file.write("visual : " + str(args.visual) + " \n") 
    file.write("train dataset : " + train_dataset + " \n") 
    file.write("test dataset : " + test_dataset + " \n") 
    s = ', '
    file.write("final test accuracy : " + str(test_accuracies[-1])  + " \n") 
    file.write(" \n") 
    file.write("epochs : " + str(args.epochs) + " \n") 
    file.write("batch size: " + str(args.batch_size) + " \n") 
    file.close()
