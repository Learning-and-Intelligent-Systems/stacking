"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

def score(model, x, k=100):
    # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

    with torch.no_grad():
        # take k monte-carlo samples of forward pass w/ dropout
        Y = torch.stack([model(x) for i in range(k)], dim=1)
        H1 = H(Y.mean(axis=1)).sum(axis=1)
        H2 = H(Y).sum(axis=(1,2))/k

        return H1 - H2
