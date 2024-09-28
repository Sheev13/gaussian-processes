"""
This file contains the following classes:
    _BaseMean
    ZeroMean
    ConstantMean
    OFSMean
    FSSMean
    
Note that for classification, desired behaviour in the constrained
[0, 1] space of class probabilities is specified by a prior mean 
function in unconstrained space, i.e. pre-sigmoid.

In other words, a mean function of f = 2x + 3 corresponds to a prior
mean in output space of logistic_function(2x + 3).
"""
import torch
from torch import nn
import warnings

from typing import Optional


class _BaseMean(nn.Module):
    """
    A base class for all prior mean functions to inherit.
    """
    def __init__(self, **redundant_kwargs):
        super().__init__()
        # logic for handling unexpected keywords. 
        redundant_kws = list(redundant_kwargs.keys())
        # 'train_mean_func' only reaches here if there are no trainable 
        # parameters in the child class
        if 'train_mean_func' in redundant_kws:
            if redundant_kwargs['train_mean_func']:
                warnings.warn("Warning: 'train_mean_func' set to True, but there are no trainable parameters for this mean function.")
            redundant_kws.remove('train_mean_func')
        if 'dims' in redundant_kws:
            redundant_kws.remove('dims')
        if len(redundant_kws) > 0:
            warnings.warn(f"Warning: unexpected kwargs passed to mean function. Proceeding by ignoring them.\nRedundant kwargs: {redundant_kws}")
    
    def forward(self, input: torch.tensor):
        raise NotImplementedError("Forward pass not implemented for _BaseMean class")
        
        

class ZeroMean(_BaseMean):
    """
    A class for a zero prior mean function:
        f = 0.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input: torch.tensor):
        return torch.zeros_like(input[:,0])
    
    
    
class ConstantMean(_BaseMean):
    """
    A class for a constant value prior mean function:
        f = c
    where c is a parameter.
    
    Args:
        prior_mean_init:
            a float representing the (initial) constant value of the prior mean.
            Default: 0.0.
        
        train_mean_func: 
            a boolean flag denoting whether or not the constant should 
            be optimised along with any other hyper and/or variational parameters.
            Default: False.
    """
        
    def __init__(self,
                 train_mean_func: bool = False,
                 prior_mean_init: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.c = nn.Parameter(torch.tensor(prior_mean_init), requires_grad=train_mean_func)
        
    def forward(self, input: torch.tensor):
        return self.c * torch.ones_like(input[:,0])