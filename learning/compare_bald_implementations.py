"""
Massachusetts Institute of Technology

Izzy Brand, 2020

Just making sure Mike and I implemented this the same way
"""

import numpy as np
import torch

# Izzy implementation
def H(x, eps=1e-7):
    """ Compute the element-wise entropy of x

    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)

    Keyword Arguments:
        eps {float} -- prevent failure on x == 0

    Returns:
        torch.Tensor -- H(x)
    """
    return -(x+eps)*torch.log(x+eps)

# Izzy impementation
def score(p, k=100):
    # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

    # computing the mutual information requires a label distribution. the
    # model predicts probility of stable p, so the distribution is p, 1-p
    Y = torch.stack([p, 1-p], axis=2) # [n x k x 2]
    # 1. average over the sample dimensions to get mean class probs
    # 2. compute the entropy of the class distribution
    H1 = H(Y.mean(axis=1)).sum(axis=1)
    # 1. compute the entropy of the sample and class distribution
    # 2. and sum over the class dimension
    # 3. and average over the sample dimsnions
    H2 = H(Y).sum(axis=(1,2))/k

    return H1 - H2

# Mike Implementation
def bald(all_preds, eps=1e-7):

    mp_c1 = torch.mean(all_preds, dim=0)
    mp_c0 = torch.mean(1 - all_preds, dim=0)

    m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))

    p_c1 = all_preds
    p_c0 = 1 - all_preds
    ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
    ent = torch.mean(ent_per_model, dim=0)

    return m_ent + ent


for _ in range(100):
    num_samples = np.random.randint(10, 1000)
    preds = torch.rand(1, num_samples)
    score1 = score(preds, k=num_samples) # takes [batchsize x num_mc_samples]
    score2 = bald(preds.T) # takes [num_mc_samplees x batch_size]
    print(f'With {num_samples} mc_dropout samples:\t{(score1-score2).item()}')
