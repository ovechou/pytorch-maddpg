import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
from torch.distributions import Categorical

n_agents = 9

def onehot_from_logits(logits, eps=0.0, explore=False):
    '''
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    '''
    # get best (according to current policy) actions in one-hot form
    argmax_acs = []
    if eps == 0.0 or explore == True:
        for i in range(n_agents):
            argmax_acs.append(torch.argmax(logits[i]))
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Categorical(logits).sample().long()
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs for i, r in enumerate(torch.rand(logits.shape[0]))])