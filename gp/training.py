"""
This file contains the following functions:
    train_variational_gp,
    train_gp.
    
This file contains the following classes:
    Minibatcher.
"""
import torch
from torch import nn
from tqdm import tqdm
from typing import Optional, List, Tuple
from collections import defaultdict
import warnings

from .utils import evaluate_gp_classifier, evaluate_gp_regressor
from .likelihoods import BernoulliLikelihood

def train_variational_gp(
    model,
    X: torch.Tensor,
    t: torch.Tensor,
    epochs: Optional[int] = None,
    training_steps: Optional[int] = None,
    learning_rate: float = 1e-2,
    num_samples: int = 1,
    batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    X_test: Optional[torch.Tensor] = None,
    t_test: Optional[torch.Tensor] = None,
    final_learning_rate: Optional[float] = None,
    max_gradient: Optional[float] = None,
    unfreeze_trainable_hypers_at_step: Optional[int] = None,
    use_gpu: bool = False,
) -> torch.Tensor:
    """
    Executes the training loop for a `SparseVariationalGaussianProcess` as defined
    in `models.py`
    
    Args:
        model:
            the user's SparseVariationalGaussianProcess instance.
        X:
            the input features of the dataset, as a torch.Tensor. If there are n 
            datapoints and d features (/feature dimensions), `X` should be of shape 
            (n, d).
        t:
            the target feature of the dataset, as a torch.Tensor. `models.py`
            currently only supports single-output GPs (i.e. regression with one 
            output or binary classification), and so `t` is expected to have shape
            (n,). The letter 't' is used instead of the more common 'y' since this
            was originally a classification-only implementation.
        epochs:
            an integer denoting the number of passes through the full dataset
            to be performed during training. Note that for small datasets, this 
            could be in the hundreds or thousands, but for large datasets with 
            millions of points this should be something like two or three, assuming
            batch size is chosen to maintain roughly constant time for an update
            step. If `training_steps` is set, `epochs` is ignored.
            Default: None.
        training_steps:
            an integer denoting the number of gradient update steps to be performed
            during training. If neither this nor `epochs` is set, `training_steps`
            defaults to 10_000, but if both are set then `training_steps` takes
            priority.
            Default: None.
        learning_rate:
            a float denoting the coefficient for gradient update steps. Typically
            this should be between 1e-4 and 1e-2.
            Default: 1e-2.
        num_samples:
            an integer denoting the number of Monte Carlo samples used to estimate
            the expected log-likelihood term within the ELBO objective function.
            A higher value results in less noisy estimates of the objective function,
            but is more costly to evaluate. It is common practice in variational
            inference to use just one sample. This is because we tend to use smaller 
            learning rates and higher numbers of training steps, meaning the noise
            in estimating the ELBO gets "averaged out" over training.
            Default: 1.
        batch_size:
            an integer denoting the number of training points to be used in a
            minibatch at each training step. This is the bread and butter of 
            stochastic gradient descent---rather than evaluating the objective
            function for the entire dataset (very expensive) at each training
            step, approximate it with the objective function evaluated at a small
            subset of the dataset (much cheaper) and treat this value as a noisy 
            (stochastic) estimate of the desired objective function evaluation.
            If this argument is left as None, no minibatching is performed, and
            the full dataset is used at every training step. For a dataset of
            size n, a batch size of m, e epochs, and N training steps, the 
            relationship is:
                N * m = n * e.
            Default: None.
        test_batch_size:
            an integer denoting the number of test points to be used in the test-
            set evaluation metrics at each training step. This should be set when
            the test set is very large to ensure that test-set evaluation metrics
            are not the computational bottleneck of the training loop. If this 
            is left as None, the full test set is used instead.
            Default: None.
        X_test: 
            the input features of the test dataset, as a torch.Tensor. If there are
            p test points and d features (/feature dimensions), `X_test` should be 
            of shape (p, d). If this argument is not specified, no test-set evaluations
            are performed during training.
            Default: None.
        t_test:
            the target feature of the dataset, as a torch.Tensor. This is expected 
            to have shape (p,). This argument *must* be specified if `X_test` has 
            been specified.
            Default: None.
        final_learning_rate:
            a float denoting the gradient update coefficient at the final training
            step. If this is left as None, `learning_rate` is used throughout training.
            If this is set, a linear learning rate scheduler is used throughout
            training, so at training step 0 the learning rate is given by 
            `learning_rate`, and at training step r out of n the learning rate is
            given by:
                learning_rate * (1 - r/n) + final_learning_rate * (r/n).
            Tempering of the learning rate is common practice in gradient-based
            optimisation for better convergence.
            Default: None.
        max_gradient:
            a float denoting the norm of the maximum gradient that can be used in
            the gradient update steps. This is to be used for numerical stability
            if noisy/exploding gradients are causing problems in training. If this 
            is set, it clips the norm of the gradient used in the update equation to
            the set value. Otherwise, no clipping is performed.
            Default: None.
        unfreeze_trainable_hypers_at_step:
            an integer denoting the training step at which to begin training any
            hyperparameters that have been set to trainable by the user. The idea
            is to initially train *just* the variational parameters to obtain a good
            posterior approximation of the inducing variables, and once they are 
            in good stead to then free the hyperparams. This tends to achieve 
            more consistent convergence than training everything from scratch
            simultaneously. If left as None, everything is trained from scratch from 
            the beginning of the training loop.
            Default: None.
        use_gpu:
            a boolean flag indicating whether to perform computations on a GPU if
            there is one available.
            Default: False.
    
    Returns:
        tracker:
            a defaultdict. The keys are the names of various useful metrics, such as 
            the expected log likelihood, ELBO, or trainable hyperparameters. The values
            are lists of corresponding values that represent the history of the metric 
            at each training step.
    """
    
    # set device to gpu if user wants to and one is available.
    if use_gpu:
        if not torch.cuda.is_available():
            print("No GPU available. Falling back to CPU.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_device(device)
        model.to(device)
        X = X.to(device)
        t = t.to(device)
        if X_test is not None and t_test is not None:
            X_test = X_test.to(device)
            t_test = t_test.to(device)
            
    # ensure batch size is valid, initialise minibatch handler.
    dataset_size = X.shape[0]
    if batch_size is None:
        batch_size = dataset_size
    if batch_size > dataset_size:
        batch_size = dataset_size
        warnings.warn("batch_size larger than dataset_size. Proceeding by setting batch_size = dataset_size")
    minibatcher = Minibatcher(dataset_size, batch_size)
    
    # handle minibatching of test data if relevant.
    if X_test is not None:
        if X_test.shape[0] > 500 and test_batch_size is None:
            warnings.warn("Warning: test-set size is greater than 500, setting test_batch_size strongly recommended!")
        if test_batch_size is None:
            test_batch_size = X_test.shape[0]
        test_minibatcher = Minibatcher(X_test.shape[0], min(test_batch_size, X_test.shape[0]))

    # set number of training steps based on user's epochs or training steps input.
    if (epochs is None) == (training_steps is None):
        if training_steps is None:
            training_steps = 10_000
        warnings.warn(f"Exactly one of `epochs` or `training_steps` must be set. Defaulting to training_steps={training_steps}")
    elif training_steps is None:
        training_steps = int(epochs * dataset_size / batch_size)

    # initialise optimiser and learning rate scheduler.
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if final_learning_rate is not None:
        end_factor = final_learning_rate / learning_rate
    else:
        end_factor = 1.0
    lr_sched = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=end_factor, total_iters=training_steps
    )

    # if hypers are to be trained later, freeze the unfrozen
    # ones and store their names in order to know what to
    # unfreeze later on.
    if unfreeze_trainable_hypers_at_step is not None:
        trainable_params = []
        for name, param in model.likelihood.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
                param.requires_grad = False
        for name, param in model.prior.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
                param.requires_grad = False

    # initialise metrics tracker.
    tracker = defaultdict(list)
    pbar = tqdm(range(training_steps))
    
    # main training loop here.
    for training_step in pbar:

        if unfreeze_trainable_hypers_at_step is not None:
            if training_step == unfreeze_trainable_hypers_at_step:
                for name, param in trainable_params:
                    param.requires_grad = True
                
        # get next minibatch.
        batch_idx = minibatcher.next_indices()
        if use_gpu:
            batch_idx.to(torch.get_default_device())
        X_batch, t_batch = X[batch_idx], t[batch_idx]

        # compute loss and gradients.
        optimiser.zero_grad()
        assert dataset_size is not None
        loss, metrics = model.loss(
            X_batch,
            t_batch,
            dataset_size,
            num_samples=num_samples,
        )
        loss.backward()
        
        # clip gradients if desired.
        if max_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient) 
        for p in model.parameters():
            if p.grad is not None:
                if p.grad.data.isnan().any():
                    p.grad.data = torch.nan_to_num(p.grad.data)
                    warnings.warn("Warning: NaN gradients encountered. Proceeded by setting them to zero.")
        
        # perform gradient update step.
        optimiser.step()
        lr_sched.step()        

        # evaluate test-set metrics if relevant.
        if X_test is not None:
            assert t_test is not None
            if test_batch_size is None:
                X_test_batch, t_test_batch = X_test, t_test
            else:
                test_batch_idx = test_minibatcher.next_indices()
                X_test_batch, t_test_batch = X_test[test_batch_idx], t_test[test_batch_idx]            

            with torch.no_grad():
                if isinstance(model.likelihood, BernoulliLikelihood):
                    test_acc = evaluate_gp_classifier(model, X_test_batch, t_test_batch)
                    metrics["test acc"] = test_acc.detach().item()
                else:
                    test_rmse = evaluate_gp_regressor(model, X_test_batch, t_test_batch, variational=True)[0]
                    metrics["test rmse"] = test_rmse.detach().item()
        
        # store metrics.
        for key, value in metrics.items():
            tracker[key].append(float(value))

        metrics["Epochs"] = (training_step * batch_size) / dataset_size

        pbar.set_postfix(metrics)
        
    # return model to cpu if relevant.
    if use_gpu and torch.cuda.is_available():
        print("Returning model and tensors to CPU.")
        cpu_device = torch.device('cpu')
        torch.set_default_device(cpu_device)
        model.to(cpu_device)

    return tracker


def train_gp(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    test_batch_size: Optional[int] = None,
    X_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    final_learning_rate: Optional[float] = None,
    max_gradient: Optional[float] = None,
    use_gpu: bool = False,
) -> torch.Tensor:
    """
    Executes the training loop for a `GaussianProcess` as defined
    in `models.py`
    
    Args:
        model:
            the user's GaussianProcess instance.
        X:
            the input features of the dataset, as a torch.Tensor. If there are n 
            datapoints and d features (/feature dimensions), `X` should be of shape 
            (n, d).
        y:
            the target feature of the dataset, as a torch.Tensor. `models.py`
            currently only supports single-output GPs, and so `y` is expected to have
            shape (n,).
        epochs:
            an integer denoting the number of passes through the full dataset
            to be performed during training. Note that minibatching is not possible
            for exact GPs.
            Default: 100.
        learning_rate:
            a float denoting the coefficient for gradient update steps. Typically
            this should be between 1e-4 and 1e-2.
            Default: 1e-2.
        test_batch_size:
            an integer denoting the number of test points to be used in the test-
            set evaluation metrics at each training step. This should be set when
            the test set is very large to ensure that test-set evaluation metrics
            are not the computational bottleneck of the training loop. If this 
            is left as None, the full test set is used instead.
            Default: None.
        X_test: 
            the input features of the test dataset, as a torch.Tensor. If there are
            p test points and d features (/feature dimensions), `X_test` should be 
            of shape (p, d). If this argument is not specified, no test-set evaluations
            are performed during training.
            Default: None.
        y_test:
            the target feature of the dataset, as a torch.Tensor. This is expected 
            to have shape (p,). This argument *must* be specified if `X_test` has 
            been specified.
            Default: None.
        final_learning_rate:
            a float denoting the gradient update coefficient at the final training
            step. If this is left as None, `learning_rate` is used throughout training.
            If this is set, a linear learning rate scheduler is used throughout
            training, so at training step 0 the learning rate is given by 
            `learning_rate`, and at training step r out of n the learning rate is
            given by:
                learning_rate * (1 - r/n) + final_learning_rate * (r/n).
            Tempering of the learning rate is common practice in gradient-based
            optimisation for better convergence.
            Default: None.
        max_gradient:
            a float denoting the norm of the maximum gradient that can be used in
            the gradient update steps. This is to be used for numerical stability
            if noisy/exploding gradients are causing problems in training. If this 
            is set, it clips the norm of the gradient used in the update equation to
            the set value. Otherwise, no clipping is performed.
            Default: None.
        use_gpu:
            a boolean flag indicating whether to perform computations on a GPU if
            there is one available.
            Default: False.
    
    Returns:
        tracker:
            a defaultdict. The keys are the names of various useful metrics, such as 
            the expected log likelihood, lml, or trainable hyperparameters. The values
            are lists of corresponding values that represent the history of the metric 
            at each training step.
    """
    # set device to gpu if user wants to and one is available.
    if use_gpu:
        if not torch.cuda.is_available():
            print("No GPU available. Falling back to CPU.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_device(device)
        model.to(device)
        X = X.to(device)
        t = t.to(device)
        if X_test is not None and t_test is not None:
            X_test = X_test.to(device)
            t_test = t_test.to(device)

    # initialise optimiser and learning rate scheduler.
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if final_learning_rate is not None:
        end_factor = final_learning_rate / learning_rate
    else:
        end_factor = 1.0
    lr_sched = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=end_factor, total_iters=epochs
    )
    
    # handle minibatching of test data if relevant.
    if X_test is not None:
        if X_test.shape[0] > 500 and test_batch_size is None:
            warnings.warn("Warning: test-set size is greater than 500, setting test_batch_size strongly recommended!")
        if test_batch_size is None:
            test_batch_size = X_test.shape[0]
        test_minibatcher = Minibatcher(X_test.shape[0], min(test_batch_size, X_test.shape[0]))

    # initialise metrics tracker.
    tracker = defaultdict(list)
    pbar = tqdm(range(epochs))
    
    # main training loop here.
    for _ in pbar:
    
        # compute loss and gradients.
        optimiser.zero_grad()
        loss, metrics = model.loss(
            X,
            y,
        )
        loss.backward()
        
        # clip gradients if desired.
        if max_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient) 
        for p in model.parameters():
            if p.grad is not None:
                if p.grad.data.isnan().any():
                    p.grad.data = torch.nan_to_num(p.grad.data)
                    warnings.warn("Warning: NaN gradients encountered. Proceeded by setting them to zero.")
        
        # perform gradient update step.
        optimiser.step()
        lr_sched.step()        

        # evaluate test-set metrics if relevant.
        if X_test is not None:
            assert y_test is not None
            if test_batch_size is None:
                X_test_batch, y_test_batch = X_test, y_test
            else:
                test_batch_idx = test_minibatcher.next_indices()
                X_test_batch, y_test_batch = X_test[test_batch_idx], y_test[test_batch_idx]            

            with torch.no_grad():
                test_rmse = evaluate_gp_regressor(model, X_test_batch, y_test_batch, X, y, variational=False)[0]
            metrics["test rmse"] = test_rmse.detach().item()
        
        # store metrics.
        for key, value in metrics.items():
            tracker[key].append(float(value))

        pbar.set_postfix(metrics)
        
    # return model to cpu if relevant.
    if use_gpu and torch.cuda.is_available():
        print("Returning model and tensors to CPU.")
        cpu_device = torch.device('cpu')
        torch.set_default_device(cpu_device)
        model.to(cpu_device)

    return tracker


class Minibatcher:
    """
    An object to handle minibatching.
    
    Args:
        dataset_size:
            an integer denoting the number of points in the dataset.
        batch_size:
            an integer denoting the number of points to be used in
            each minibatch.
    """
    def __init__(self, dataset_size, batch_size):
        self.N = dataset_size
        self.M = batch_size
        self.steps_per_epoch = int(self.N / self.M)
        self.current_step = 0
        self.indices = torch.randperm(self.N)

    def next_indices(self):
        """Returns the indices of the next minibatch such that each datapoint
        will be used exactly once per epoch (i.e. choose without replacement).
        """
        start = self.current_step * self.M
        if (self.current_step + 1) * self.M >= self.N:
            a = self.indices[start:]
            filler = self.indices[:start][:self.M-a.shape[0]]
            self.current_step = 0
            self.shuffle()
            return torch.cat((a, filler))
        else:
            end = (self.current_step + 1) * self.M
            self.current_step += 1
            return self.indices[start:end]
        
    def shuffle(self):
        self.indices = torch.randperm(self.N)