"""
This file contains the following classes:
    _BaseMaternKernel
    ExponentialKernel
    Matern1_5Kernel
    Matern2_5Kernel
    SquaredExponentialKernel
"""
import torch
import torch.nn as nn
import warnings

from typing import Optional


class _BaseMaternKernel(nn.Module):
    """
    A base class for Matern kernel classes to inherit to avoid repeated code.
    

    Args:
        num_inputs:
            an integer representing the dimensionality of the input features.
            (i.e. how many input features there are).
            Default: None
        
        l: 
            a float representing the lengthscale parameter. If ARD is being
            used, this represents the lengthscale across all dimensions.
            Default: 1.0
            
        train_l:
            a boolean flag denoting whether or not the lengthscale(s) should 
            be optimised along with any other hyper and/or variational parameters.
            Default: False
            
        fixed_ls:
            an optional argument that contains a dictionary of feature index
            (key) lengthscale (value) pairs that are to be held fixed if ARD 
            is being used.
            Default: None
            
        ard:
            a boolean flag denoting whether or not to have different lengthscales
            for different feature dimensions. This is only useful if `fixed_ls` 
            True so that different lengthscales can be learned.
            Default: False
    """
    def __init__(self,
                 num_inputs: Optional[int] = None,
                 l: float = 1.0,
                 train_l: bool = False,
                 fixed_ls: Optional[dict] = None,
                 ard: bool = False):
        super().__init__()
        # initialise lengthscale parameter. We parameterise it in log-space since 
        # lengthscales have to be positive.
        if ard:
            if num_inputs is None:
                raise ValueError("ARD set to True, but num_inputs not specified")
            self.log_l = nn.Parameter((torch.ones((num_inputs,)) * l).log(), requires_grad=train_l)
        else:
            self.log_l = nn.Parameter(torch.tensor(l).log(), requires_grad=train_l)
            if fixed_ls is not None:
                warnings.warn("Warning: guideprice lengthscale provided as well as one-size-fits-all lengthscale.\nProceeding by ignoring guideprice lengthscale")
        self.ard = ard
        self.fixed_ls = fixed_ls
    
    @property
    def l(self):
        # clamp here is to avoid division by zero numerical instabilities
        ls = self.log_l.exp().clamp(min=1e-8)
        if self.fixed_ls is not None:
            for dim, l in self.fixed_ls.items():
                ls[dim] = l
        return ls

    def compute_distances(self, x1: torch.Tensor, x2: torch.Tensor):
        x1, x2 = x1 / self.l, x2/ self.l
        return torch.cdist(x1, x2, p=2).clamp(min=0.0)
    
    def diagonal(self, x1: torch.Tensor):
        # sometimes we only need the diagonal of a covariance matrix
        return torch.ones_like(x1[:,0])
    
    def k(self, d: torch.Tensor):
        raise NotImplementedError("Kernel function not implemented for _BaseMaternKernel class")

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None):
        # compute the covariance matrix between two feature matrices
        if x2 is None:
            # the two feature matrices are x1 and x1
            d = self.compute_distances(x1, x1)
            # add a small amount of jitter (identity matrix scaled down) for numerical stability
            return self.k(d) + torch.eye(x1.shape[0]) * 1e-8
        else:
            # the two feature matrices are x1 and x2
            d = self.compute_distances(x1, x2)
            return self.k(d)
                                    


class ExponentialKernel(_BaseMaternKernel):
    """
    The exponential kernel. Matern kernel with nu=0.5.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def k(self, d: torch.Tensor):
        return torch.exp(-d)



class Matern1_5Kernel(_BaseMaternKernel):
    """
    The Matern kernel with nu=1.5.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def k(self, d: torch.Tensor):
        return (1 + torch.tensor(3.0).sqrt() * d) * torch.exp(-torch.tensor(3.0).sqrt() * d)
    
    
    
class Matern2_5Kernel(_BaseMaternKernel):
    """
    The Matern kernel with nu=2.5
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def k(self, d: torch.Tensor):
        return (1 + torch.tensor(5.0).sqrt() * d + (5/3) * d.square()) * torch.exp(-torch.tensor(5.0).sqrt() * d)



class SquaredExponentialKernel(_BaseMaternKernel):
    """
    The squared exponential kernel. Gaussian radial basis function (RBF) kernel.
    Exponentiated quadratic kernel. Matern kernel with nu->\infty.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def k(self, d: torch.Tensor):
        return torch.exp(-0.5 * d.square())