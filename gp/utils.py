"""
This file contains the following functions:
    evaluate_gp_classifier,
    evaluate_gp_regressor,
    get_predictive_dist_grid,
    choose_m_from_n
"""
import torch
from torch import nn
from typing import Optional, List


def evaluate_gp_classifier(
    model,
    X_test: torch.Tensor,
    t_test: torch.Tensor,
    X: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
):
    """
    Evaluation function for a SparseVariationalGaussianProcess
    object with a 'Bernoulli' likelihood.
    
    Args:
        model:
            the model instance.
        X_test:
            the test inputs.
        t_test:
            the test outputs.
        X:
            the training inputs.
            Default: None.
        t: 
            the training inputs.
            Default: None.
            
    Returns:
        accuracy:
            the percentage of the test points for which the model
            made the correct binary classification.
        elbo:
            an estimate of the ELBO of a large (capped at 100k points)
            number of training points. This is only returned if X and 
            t are supplied and not left as None.
    """

    with torch.no_grad():
        predictive_probs = model(X_test).probs

    class_predictions = predictive_probs.round()
    accuracy = 1 - (class_predictions - t_test).abs().mean()
    
    if X is not None:
        assert t is not None        
        batch_size = min(X.shape[0], 100_000)
        batch_idx = choose_m_from_n(X.shape[0], batch_size)[0]
        X_batch, t_batch = X[batch_idx], t[batch_idx]   
        with torch.no_grad():
            elbo = - model.loss(X_batch, t_batch, X.shape[0], num_samples=100)[0]
        return accuracy, elbo
    
    return accuracy

def evaluate_gp_regressor(
    model,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    X: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    variational: bool = False,
):
    """
    Evaluation function for a SparseVariationalGaussianProcess
    or Gaussian Process object  object with a 'Gaussian' likelihood.
    
    Args:
        model:
            the model instance.
        X_test:
            the test inputs.
        t_test:
            the test outputs.
        X:
            the training inputs.
            Default: None.
        y: 
            the training inputs.
            Default: None.
        variational:
            a boolean flag indicating whether a variational GP is
            being used or an exact GP.
            Default: False.
            
    Returns:
        rmse:
            the root mean squared error of the model predictive mean over the test points.
        elbo:
            an estimate of the ELBO of a large (capped at 100k points)
            number of training points. This is only returned if X and 
            y are supplied and not left as None, and if variational is True.
        lml:
            the log marginal likelihood evaluated on the training points. This is
            only returned if X and y are supplied and not left as None, and if
            variational is False.
    """
    if variational:
        with torch.no_grad():
            predictive_probs = model(X_test)

        rmse = (predictive_probs.mean - y_test).pow(2).mean().sqrt()
        
        if X is not None:
            assert y is not None        
            batch_size = min(X.shape[0], 100_000)
            batch_idx = choose_m_from_n(X.shape[0], batch_size)[0]
            X_batch, y_batch = X[batch_idx], y[batch_idx]   
            with torch.no_grad():
                elbo = - model.loss(X_batch, y_batch, X.shape[0], num_samples=100)[0]
            return rmse, elbo
        
        return rmse, None
        
    else:
        with torch.no_grad():
            predictive_probs = model(X_test, X, y)

        rmse = (predictive_probs.mean - y_test).pow(2).mean().sqrt()
        with torch.no_grad():
            lml = - model.loss(X, y)[0]
        return rmse, lml


def get_predictive_dist_grid(
    model,
    X: torch.Tensor,
    x1lim: List[int],
    x2lim: List[int],
    granularity: int = 100,
):
    # For 2D classification visualisations such as with the Banana dataset
    
    x1_range = torch.linspace(x1lim[0], x1lim[1], granularity)
    x2_range = torch.linspace(x2lim[0], x2lim[1], granularity)
    xx1, xx2 = torch.meshgrid(x1_range, x2_range)
    
    X_eval = torch.cat((xx1.flatten().unsqueeze(1), xx2.flatten().unsqueeze(1)), dim=1)
    
    assert X_eval.shape[0] == granularity ** 2
    assert X_eval.shape[1] == 2

    with torch.no_grad():
        predictive_probs = model(X_eval).probs
        
    pred_probs_grid = predictive_probs.reshape_as(xx1)
    
    return xx1, xx2, pred_probs_grid


def choose_m_from_n(n: int, m: int):
    assert m <= n
    idx = torch.randperm(n)
    return idx[:m], idx[m:]