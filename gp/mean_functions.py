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
    
    
    
class OFSMean(_BaseMean):
    """
    A class for a bespoke offer-select prior mean function:
        f = m(x - c)
    where x is the guideprice feature, m is a strictly positive gradient,
    c is some offset parameter.
    
    Args:
        positive_gradient_init:
            a float representing the (initial) value of m.
            Default: 1.0.
            
        offset_init:
            a float representing the (initial) value of c.
            Default: 2.0.
            
        guideprice_dim:
            the index of the guideprice dimension.
            Default: None.
        
        train_mean_func: 
            a boolean flag denoting whether or not the constant should 
            be optimised along with any other hyper and/or variational parameters.
            Default: False.
    """
    def __init__(self,
                 train_mean_func: bool = False,
                 guideprice_dim: Optional[int] = None,
                 positive_gradient_init: float = 1.0,
                 offset_init: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)
        # parameterise m in log space to enforce positivity. Also take absolute 
        # value of user-passed value to ensure positivity there.
        self.log_m = nn.Parameter(torch.tensor(positive_gradient_init).abs().log(), requires_grad=train_mean_func)
        self.offset = nn.Parameter(torch.tensor(offset_init), requires_grad=train_mean_func)
        if guideprice_dim is None:
            warnings.warn("Warning: no guide price dimension has been set. Defaulting to guideprice_dim = 0.")
            guideprice_dim = 0
        self.guideprice_dim = guideprice_dim
        
    @property
    def m(self):
        return self.log_m.exp()
        
    def forward(self, input: torch.tensor):
        return self.m * (input[:, self.guideprice_dim] - self.offset) # = m(x - c)


class FSSMean(_BaseMean):
    """
    A class for a bespoke for-sale-to-sold prior mean function:
        f = -s(x-l)^2 + p
    where s, l, and p are sharpness, location, and peak parameters
    respectively, and where x is the guideprice feature. s is 
    forced to be positive.
    
    Args:
        sharpness_init:
            a float representing the (initial) value of s.
            Default: 1.0.
            
        peak_init:
            a float representing the (initial) value of p.
            Default: 0.0.
        
        loc_init:
            a float representing the (initial) value of l.
            Default: 0.0.
            
        guideprice_dim:
            the index of the guideprice dimension.
            Default: None.
        
        train_mean_func: 
            a boolean flag denoting whether or not the constant should 
            be optimised along with any other hyper and/or variational parameters.
            Default: False.
    """
    def __init__(self,
                 train_mean_func: bool = False,
                 guideprice_dim: Optional[int] = None,
                 sharpness_init: float = 1.0,
                 peak_init: float = 0.0,
                 loc_init: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        if sharpness_init < 0.0:
            warnings.warn(f"Warning: sharpness must be a positive float, but {sharpness_init} was provided. Proceeding by setting to {-sharpness_init}")
            sharpness_init = -sharpness_init
        if sharpness_init < 1e-4:
            sharpness_init = 1e-4
        # parameterise s in log-space to enforce positivity
        self.log_sharpness = nn.Parameter(torch.tensor(sharpness_init).log(), requires_grad=train_mean_func)
        self.peak = nn.Parameter(torch.tensor(peak_init), requires_grad=train_mean_func)
        self.loc = nn.Parameter(torch.tensor(loc_init), requires_grad=train_mean_func)
        if guideprice_dim is None:
            warnings.warn("Warning: no guide price dimension has been set. Defaulting to guideprice_dim = 0.")
            guideprice_dim = 0
        self.guideprice_dim = guideprice_dim
        
    @property
    def sharpness(self):
        return self.log_sharpness.exp()
    
    def forward(self, input: torch.tensor):
        return - self.sharpness * (input[:, self.guideprice_dim] - self.loc).pow(2) + self.peak