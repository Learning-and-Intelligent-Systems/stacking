import copy
import numpy as np
import pickle
import torch

from itertools import islice
from torch.utils.data import Dataset, DataLoader, Sampler
    
class ABCBlocksTransDataset(Dataset):
    def __init__(self):
        self.states = torch.tensor([])
        self.actions = torch.tensor([])
        self.next_states = torch.tensor([])

    def __getitem__(self, ix):
        return [self.states[ix], 
                self.actions[ix]], \
                self.next_states[ix]
        
    def __len__(self):
        """
        The total number of datapoints in the entire dataset.
        """
        return len(self.states)    

    def remove_elements(self, remove_list):
        mask = np.ones(len(self.states), dtype=bool)
        mask[remove_list] = False
        self.states = self.states[mask]
        self.actions = self.actions[mask]
        self.next_states = self.next_states[mask]

    def add_to_dataset(self, state, action, next_state):
        self.states = torch.cat([self.states, torch.tensor([state])])
        self.actions = torch.cat([self.actions, torch.tensor([action])])
        self.next_states = torch.cat([self.next_states, torch.tensor([next_state])])
        
    
class ABCBlocksHeurDataset(Dataset):
    def __init__(self):
        self.states = torch.tensor([])
        self.goals = torch.tensor([])
        self.steps_to_goal = torch.tensor([], dtype=torch.float64)

    def __getitem__(self, ix):
        return [self.states[ix], 
                self.goals[ix]], \
                self.steps_to_goal[ix]
        
    def __len__(self):
        """
        The total number of datapoints in the entire dataset.
        """
        return len(self.states)    

    def add_to_dataset(self, state, goal, steps_to_goal):
        self.states = torch.cat([self.states, torch.tensor([state])])
        self.goals = torch.cat([self.goals, torch.tensor([goal])])
        self.steps_to_goal = torch.cat([self.steps_to_goal, torch.tensor([steps_to_goal])])