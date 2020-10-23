import torch
from torch import nn

class Resetable(nn.Module):
    def __init__(self):
        self.backup_state_dict = None
        super(Resetable, self).__init__()

    def backup(self):
        self.backup_state_dict = self.state_dict().copy()

    def reset(self):
        if self.backup_state_dict is not None:
            self.load_state_dict(self.backup_state_dict)
