import copy
import numpy as np
import pickle
import torch

from itertools import islice
from torch.utils.data import Dataset, DataLoader, Sampler
from learning.models.goal_conditioned import TransitionGNN, HeuristicGNN

class ABCBlocksTransDataset(Dataset):
    def __init__(self):
        self.object_features = torch.tensor([], dtype=torch.float64)
        self.edge_features = torch.tensor([], dtype=torch.float64)
        self.actions = torch.tensor([], dtype=torch.float64)
        self.delta_edge_features = torch.tensor([], dtype=torch.float64)
        self.next_edge_features = torch.tensor([], dtype=torch.float64)
        self.optimistic_accuracy = torch.tensor([], dtype=torch.float64)

    def set_pred_type(self, pred_type):
        self.pred_type = pred_type

    def __getitem__(self, ix):
        assert(self.pred_type, 'Must set pred_type to getitem from ABCBlocksTransDataset')
        if self.pred_type == 'full_state':
            return [self.object_features[ix],
                    self.edge_features[ix],
                    self.actions[ix]], \
                    self.next_edge_features[ix]
        elif self.pred_type == 'delta_state':
            return [self.object_features[ix],
                    self.edge_features[ix],
                    self.actions[ix]], \
                    self.delta_edge_features[ix]
        elif self.pred_type == 'class':
            return [self.object_features[ix],
                    self.edge_features[ix],
                    self.actions[ix]], \
                    self.optimistic_accuracy[ix]

    def __len__(self):
        """
        The total number of datapoints in the entire dataset.
        """
        return len(self.object_features)

    def remove_elements(self, remove_list):
        mask = np.ones(len(self.edge_features), dtype=bool)
        mask[remove_list] = False
        self.object_features = self.object_features[mask]
        self.edge_features = self.edge_features[mask]
        self.actions = self.actions[mask]
        self.next_edge_features = self.next_edge_features[mask]
        self.delta_edge_features = self.delta_edge_features[mask]
        self.optimistic_accuracy = self.optimistic_accuracy[mask]

    def add_to_dataset(self, object_features, edge_features, action, next_edge_features, delta_edge_features, optimistic_accuracy):
        self.object_features = torch.cat([self.object_features, torch.tensor([object_features])])
        self.edge_features = torch.cat([self.edge_features, torch.tensor([edge_features])])
        self.actions = torch.cat([self.actions, torch.tensor([action])])
        self.next_edge_features = torch.cat([self.next_edge_features, torch.tensor([next_edge_features])])
        self.delta_edge_features = torch.cat([self.delta_edge_features, torch.tensor([delta_edge_features])])
        self.optimistic_accuracy = torch.cat([self.optimistic_accuracy, torch.tensor([optimistic_accuracy])])

class ABCBlocksHeurDataset(Dataset):
    def __init__(self):
        self.object_features = torch.tensor([], dtype=torch.float64)
        self.edge_features = torch.tensor([], dtype=torch.float64)
        self.goal_edge_features = torch.tensor([], dtype=torch.float64)
        self.steps_to_goal = torch.tensor([], dtype=torch.float64)

    def __getitem__(self, ix):
        return [self.object_features[ix],
                self.edge_features[ix],
                self.goal_edge_features[ix]], \
                self.steps_to_goal[ix]

    def __len__(self):
        """
        The total number of datapoints in the entire dataset.
        """
        return len(self.object_features)

    def add_to_dataset(self, object_features, edge_features, goal_edge_features, steps_to_goal):
        self.object_features = torch.cat([self.object_features, torch.tensor([object_features])])
        self.edge_features = torch.cat([self.edge_features, torch.tensor([edge_features])])
        self.goal_edge_features = torch.cat([self.goal_edge_features, torch.tensor([goal_edge_features])])
        self.steps_to_goal = torch.cat([self.steps_to_goal, torch.tensor([steps_to_goal])])

# this is only for single inputs (not batches)
# TODO: have it detect if a single input or a batch is being passed in
def model_forward(model, inputs):
    tensor_inputs = []
    if isinstance(model, TransitionGNN):
        if model.pred_type != 'class':
            batch_shape_lens = [3, 4, 2]
        else:
            batch_shape_lens = [3, 4, 2]
    elif isinstance(model, HeuristicGNN):
        batch_shape_lens = [3, 4, 4]
    for batch_input_shape_len, input in zip(batch_shape_lens, inputs):
        if not torch.is_tensor(input):
            input = torch.tensor(input, dtype=torch.float64)
        if len(input.shape) == batch_input_shape_len-1: # if this is a batch of 1, then add a dimension
            input = input[None, :]
        tensor_inputs.append(input)
    output = model.forward(tensor_inputs)
    return output.detach().numpy()


# for testing
def preprocess(args, dataset, type='successful_actions'):
    xs, ys = dataset[:]
    remove_list = []
    # only keep samples with successful actions/edge changes
    if type == 'successful_actions':
        for i, ((object_features, edge_features, action), next_edge_features) in enumerate(dataset):
            if (args.pred_type == 'full_state' and (edge_features == next_edge_features).all()) or \
                (args.pred_type == 'delta_state' and (next_edge_features.abs().sum() == 0)):
                remove_list.append(i)
    # all actions have same frequency in the dataset
    if type == 'balanced_actions':
        distinct_actions = []
        actions_counter = {}
        for i, ((object_features, edge_features, action), next_edge_features) in enumerate(dataset):
            a = tuple(action.numpy())
            if a not in distinct_actions:
                distinct_actions.append(a)
                actions_counter[a] = [i]
            else:
                actions_counter[a] += [i]
        min_distinct_actions = min([len(counter) for counter in actions_counter.values()])
        for a in distinct_actions:
            remove_list += actions_counter[a][min_distinct_actions:]

    dataset.remove_elements(remove_list)
