import copy
import numpy as np
import pickle
import torch

from itertools import islice
from torch.utils.data import Dataset, DataLoader, Sampler
    
class ABCBlocksTransDataset(Dataset):
    def __init__(self):
        self.object_features = torch.tensor([])
        self.edge_features = torch.tensor([])
        self.actions = torch.tensor([])
        self.next_edge_features = torch.tensor([])

    def __getitem__(self, ix):
        return [self.object_features[ix],
                self.edge_features[ix],
                self.actions[ix]], \
                self.next_edge_features[ix]
        
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

    def add_to_dataset(self, object_features, edge_features, action, next_edge_features):
        self.object_features = torch.cat([self.object_features, torch.tensor([object_features])])
        self.edge_features = torch.cat([self.edge_features, torch.tensor([edge_features])])
        self.actions = torch.cat([self.actions, torch.tensor([action])])
        self.next_edge_features = torch.cat([self.next_edge_features, torch.tensor([next_edge_features])])
        
    
class ABCBlocksHeurDataset(Dataset):
    def __init__(self):
        self.object_features = torch.tensor([])
        self.edge_features = torch.tensor([])
        self.goal_edge_features = torch.tensor([])
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