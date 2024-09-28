"""
This file contains the following classes:
    GPPrior
    GaussianProcess
    SparseVariationalGaussianProcess
    SparseOrthogonalVariationalGaussianProcess
"""
import torch
from torch import nn
from torch import sigmoid as logistic

from .utils import choose_m_from_n
from .covariance_functions import ExponentialKernel, Matern1_5Kernel, Matern2_5Kernel, SquaredExponentialKernel
from .mean_functions import ZeroMean, ConstantMean, OFSMean, FSSMean
from .likelihoods import GaussianLikelihood, BernoulliLikelihood

from typing import Optional
import warnings


class GPPrior(nn.Module):
    """Represents a Gaussian Process Prior.
    
    Args:
        num_inputs: 
            an integer denoting the number of input dimensions.
        covariance_function:
            a string denoting the choice of covariance function. Options are:
                'exponential',
                'matern-1.5',
                'matern-2.5',
                'squared-exponential'.
            Default: 'squared-exponential'.
        mean_function:
            a string denoting the choice of prior mean function. Options are:
                'zero',
                'constant',
                'ofs',
                'fss'.
            Default: 'zero'.
        l: 
            a positive float representing the lengthscale hyperparameter of the 
            covariance function. If ARD is being used, this is the (initial) 
            lengthscale for every dimension, unless any fixed lengthscales are 
            specified via `fixed_ls`.
            Default: 1.0.
        train_l:
            a boolean flag denoting whether or not the lengthscale(s) should 
            be optimised along with any other hyper and/or variational parameters.
            Default: False.
        fixed_ls:
            an optional argument that contains a dictionary of feature index
            (key) lengthscale (value) pairs that are to be held fixed if ARD 
            is being used.
            Default: None.
        ard:
            a boolean flag denoting whether or not to have different lengthscales
            for different feature dimensions. This is only useful if `fixed_ls` 
            True so that different lengthscales can be learned.
            Default: False
        **mean_func_kwargs: 
            These are further keyword arguments that are passed to the prior mean
            function object. See `mean_functions.py` for more details.
            
            train_mean_func: 
                a boolean flag denoting whether or not any prior mean function
                parameters should be optimised along with any other hyper an/or
                variational paramters. This is ignored for mean functions that
                have no parameters (e.g. ZeroMean/'zero').
                Default: False
            prior_mean_init: 
                a float representing the (initial) constant value of the prior mean
                if the mean function is 'constant'. This is ignored if the mean
                function is something other than 'constant'.
                Default: 0.0.
            positive_gradient_init:
                a float representing the (initial) value of m. This is ignored
                if the mean function is something other than 'ofs'.
                Default: 1.0.
            offset_init:
                a float representing the (initial) value of c. This is ignored
                if the mean function is something other than 'ofs'.
                Default: 2.0.
            guideprice_dim:
                the index of the guideprice dimension. This is ignored if the 
                mean function is not 'ofs' or 'fss'.
                Default: None.
            sharpness_init:
                a float representing the (initial) value of s. This is ignored
                if the mean function is not 'fss'.
                Default: 1.0.
            peak_init:
                a float representing the (initial) value of p. This is ignored
                if the mean function is not 'fss'.
                Default: 0.0.
            loc_init:
                a float representing the (initial) value of l. This is ignored
                if the mean function is not 'fss'.
                Default: 0.0.
    """
    
    def __init__(
        self,
        num_inputs: int,
        covariance_function: str = 'squared-exponential',
        mean_function: str = 'zero',
        l: float = 1.0,
        train_l: bool = False,
        fixed_ls: Optional[dict] = None,
        ard: bool = False,
        **mean_func_kwargs,
    ):
        super().__init__()
        
        # initialise covariance function object
        covariance_function = covariance_function.lower()
        implemented_covfunc_names = ['exponential', 'matern-1.5', 'matern-2.5', 'squared-exponential']
        implemented_covfunc_objs = [ExponentialKernel, Matern1_5Kernel, Matern2_5Kernel, SquaredExponentialKernel]
        if covariance_function not in implemented_covfunc_names:
            raise NotImplementedError(f"{covariance_function} either contains a typo or corresponds to a covariance function not yet implemented")
        for i in range(len(implemented_covfunc_names)):
            if covariance_function == implemented_covfunc_names[i]:
                self.kernel = implemented_covfunc_objs[i](num_inputs=num_inputs,
                                                          l=l,               
                                                          train_l=train_l,
                                                          fixed_ls=fixed_ls,
                                                          ard=ard)
        # initialise mean function object
        mean_function = mean_function.lower()
        implemented_meanfunc_names = ['zero', 'constant', 'ofs', 'fss']
        implemented_meanfunc_objs = [ZeroMean, ConstantMean, OFSMean, FSSMean]
        if mean_function not in implemented_meanfunc_names:
            raise NotImplementedError(f"{mean_function} either contains a typo or corresponds to a mean function not yet implemented")
        for i in range(len(implemented_meanfunc_names)):
            if mean_function == implemented_meanfunc_names[i]:
                self.mean = implemented_meanfunc_objs[i](**mean_func_kwargs)
            
    def forward(self, inputs: torch.tensor):            
        mu = self.mean(inputs)
        cov = self.kernel(inputs)
        return torch.distributions.MultivariateNormal(mu, cov)
    

class GaussianProcess(nn.Module):
    """Represents an exact Gaussian Process.
    
    The implementation closely follows 
        'Gaussian Processes for Machine Learning'
        Rasmussen and Williams (2006).

    This is a regression model that can be used on small datasets.

    Args:
        num_inputs: 
            an integer denoting the number of input dimensions.
        sigma_y:
            a positive float denoting the observation noise/std of the Gaussian
            likelihood.
            Default: 0.01.
        train_sigma_y:
            a boolean flag denoting whether or not sigma_y should be optimised 
            along with any other hyper and/or variational parameters.
            Default: False.
        **prior_params:
            These are further keyword arguments that are passed to the GP prior
            object. See `GPPrior` for more details.
                
            covariance_function:
                a string denoting the choice of covariance function. Options are:
                    'exponential',
                    'matern-1.5',
                    'matern-2.5',
                    'squared-exponential'.
                Default: 'squared-exponential'.
            mean_function:
                a string denoting the choice of prior mean function. Options are:
                    'zero',
                    'constant',
                    'ofs',
                    'fss'.
                Default: 'zero'.
            l: 
                a positive float representing the lengthscale hyperparameter of the 
                covariance function. If ARD is being used, this is the (initial) 
                lengthscale for every dimension, unless any fixed lengthscales are 
                specified via `fixed_ls`.
                Default: 1.0.
            train_l:
                a boolean flag denoting whether or not the lengthscale(s) should 
                be optimised along with any other hyper and/or variational parameters.
                Default: False.
            fixed_ls:
                an optional argument that contains a dictionary of feature index
                (key) lengthscale (value) pairs that are to be held fixed if ARD 
                is being used.
                Default: None.
            ard:
                a boolean flag denoting whether or not to have different lengthscales
                for different feature dimensions. This is only useful if `fixed_ls` 
                True so that different lengthscales can be learned.
                Default: False
            **mean_func_kwargs: 
                These are further keyword arguments that are passed to the prior mean
                function object. See `mean_functions.py` for more details.

                train_mean_func: 
                    a boolean flag denoting whether or not any prior mean function
                    parameters should be optimised along with any other hyper an/or
                    variational paramters. This is ignored for mean functions that
                    have no parameters (e.g. ZeroMean/'zero').
                    Default: False
                prior_mean_init: 
                    a float representing the (initial) constant value of the prior mean
                    if the mean function is 'constant'. This is ignored if the mean
                    function is something other than 'constant'.
                    Default: 0.0.
                positive_gradient_init:
                    a float representing the (initial) value of m. This is ignored
                    if the mean function is something other than 'ofs'.
                    Default: 1.0.
                offset_init:
                    a float representing the (initial) value of c. This is ignored
                    if the mean function is something other than 'ofs'.
                    Default: 2.0.
                guideprice_dim:
                    the index of the guideprice dimension. This is ignored if the 
                    mean function is not 'ofs' or 'fss'.
                    Default: None.
                sharpness_init:
                    a float representing the (initial) value of s. This is ignored
                    if the mean function is not 'fss'.
                    Default: 1.0.
                peak_init:
                    a float representing the (initial) value of p. This is ignored
                    if the mean function is not 'fss'.
                    Default: 0.0.
                loc_init:
                    a float representing the (initial) value of l. This is ignored
                    if the mean function is not 'fss'.
                    Default: 0.0.
        
        
    Example usage:
    
        >>> import gp
        
        # to initialise a GP:
        >>> geepee = gp.models.GaussianProcess(2, sigma_y=0.1, train_sigma_y=True, ard=True, train_l=True) 
        
        # to train the GP, see `training.py`.
        
        # to obtain a collection of n prior samples:
        >>> my_test_points = torch.randn((50, 2))
        >>> samps = geepee(my_test_points).sample((n,))
        
        # to obtain (marginal) prior predictive mean and stds:
        >>> prior_dist = geepee(my_test_points)
        >>> prior_means = prior_dist.mean
        >>> prior_stds = prior_dist.variance.sqrt()
        
        # to obtain a collection of n posterior predictive samples:
        >>> samps = geepee(my_test_points, X=my_X, y=my_y).sample((n,))
        
        # to obtain (marginal) posterior predictive mean and stds:
        >>> post_dist = geepee(my_test_points, X=my_X, y=my_y)
        >>> post_means = post_dist.mean
        >>> post_stds = post_dist.variance.sqrt()
        
        # to evaluate the log marginal likelihood:
        >>> neg_lml, _ = geepee.loss(my_X, my_y)
    """

    def __init__(
        self,
        num_inputs: int,
        sigma_y: float = 0.01,
        train_sigma_y: bool = False,
        **prior_params,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.prior = GPPrior(num_inputs, **prior_params)
        self.likelihood = GaussianLikelihood(sigma_y=sigma_y, train_sigma_y=train_sigma_y)

    def p_fn(self, X_test: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
        """returns the posterior distribution over functions evaluated at X_test
        i.e. p(f(X_test)|D). This implements standard GP posterior distribution
        equations that can be found in e.g. Rasmussen & Williams 2006.
        """
        K_nn = self.prior.kernel(X) + self.likelihood.sigma_y.pow(2) * torch.eye(X.shape[0])
        L_nn = torch.linalg.cholesky(K_nn)
        K_nn_inv = torch.cholesky_inverse(L_nn)

        K_tn = self.prior.kernel(X_test, X)
        K_tt = self.prior.kernel(X_test)

        mu = K_tn @ K_nn_inv @ y
        covar = K_tt - (K_tn @ K_nn_inv @ K_tn.T)
        return torch.distributions.MultivariateNormal(loc=mu.squeeze(), covariance_matrix=covar)

    def forward(self, X_test: torch.Tensor = None, X: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None):
        """the primary prediction function of the GP for users. X_test specifies the
        inputs at which to obtain predictions. X and y are the inputs and outputs 
        respectively in the dataset. If they are not specified, e.g. left as None,
        this function returns the GP prior distribution over the test points. Otherwise,
        it returns the posterior predictive distribution over the test points.
        """
        if X is None:
            return self.prior(X_test) # useful if the user wants to sample from GP prior
        assert y is not None
        p_fn_test = self.p_fn(X_test, X, y)
        return self.likelihood.posterior_predictive(p_fn_test)

    def loss(self, X, y):
        """Computes the log marginal likelihood via standard equations that can be
        found in e.g. Rasmussen & Williams 2006. Since torch optimisers do gradient
        *descent*, this returns the *negative* log marginal likelihood. It also
        returns a dictionary of useful metrics including the log marginal likelihood
        and any trainable hyperparameters.
        """
        # objective function is the marginal likelihood
        K_nn = self.prior.kernel(X)
        chol = torch.linalg.cholesky(K_nn + self.likelihood.sigma_y**2*torch.eye(X.shape[0]))
        inv = torch.cholesky_inverse(chol)

        a = -0.5 * y.T@inv@y
        b = -0.5 * torch.linalg.det(inv).pow(-1).log()
        c = - X.shape[0]/2 * (torch.tensor(2) * torch.pi).log()
        lml = (a + b + c).squeeze()

        metrics = {
            "lml": lml.detach().item(),
        }
        
        # obtain trainable hyperparams and their values.
        # Many of these are implemented in log-space, so
        # we need to transform them into regular space.
        for name, param in self.prior.named_parameters():
            if param.requires_grad:
                if (self.prior.kernel.ard and name == 'kernel.log_l'):
                    pass
                elif 'log' in name:
                    if 'log_' in name:
                        name = name.replace("log_", "")
                    elif '_log' in name:
                        name = name.replace("_log", "")
                    else:
                        name = name.replace("log", "")
                    metrics[name] = param.exp().detach().item()
                else:
                    metrics[name] = param.detach().item()

        for name, param in self.likelihood.named_parameters():
            if param.requires_grad:
                if 'log' in name:
                    if 'log_' in name:
                        name = name.replace("log_", "")
                    elif '_log' in name:
                        name = name.replace("_log", "")
                    else:
                        name = name.replace("log", "")
                    metrics[name] = param.exp().detach().item()
                else:
                    metrics[name] = param.detach().item()
            
        return - lml, metrics
    

    

class SparseVariationalGaussianProcess(nn.Module):
    """Represents a Sparse Variational Gaussian Process.
    
    The implementation closely follows
        'Scalable Variational Gaussian Process Classification'
        Hensman, Matthews, and Ghahramani (2015)
    including variable names.

    For regression, likelihood must be Gaussian.
    For classification, likelihood must be Bernoulli.
    
    Args:
        num_inputs: 
            an integer denoting the number of input dimensions.
        num_inducing:
            an integer denoting the number of inducing points to be used. The 
            higher the better, but the more computationally expensive. To determine
            how many should be used, consider how many points would be needed to
            summarise the dataset space.
        likelihood:
            a string denoting the choice of likelihood function. Options are:
                'Gaussian': to be used for regression,
                'Bernoulli': to be used for binary classification. 
            Note that character case is ignored.
        sigma_y: 
            a positive float denoting the observation noise/std of the Gaussian
            likelihood. This is ignored unless the likelihood is 'Gaussian'.
            Default: 1e-2.
        train_sigma_y:
            a boolean flag denoting whether or not sigma_y should be optimised 
            along with any other hyper and/or variational parameters. This is 
            ignored unless the likelihood is 'Gaussian'.
            Default: False.
        **prior_params:
            These are further keyword arguments that are passed to the GP prior
            object. See `GPPrior` for more details.
                
            covariance_function:
                a string denoting the choice of covariance function. Options are:
                    'exponential',
                    'matern-1.5',
                    'matern-2.5',
                    'squared-exponential'.
                Default: 'squared-exponential'.
            mean_function:
                a string denoting the choice of prior mean function. Options are:
                    'zero',
                    'constant',
                    'ofs',
                    'fss'.
                Default: 'zero'.
            l: 
                a positive float representing the lengthscale hyperparameter of the 
                covariance function. If ARD is being used, this is the (initial) 
                lengthscale for every dimension, unless any fixed lengthscales are 
                specified via `fixed_ls`.
                Default: 1.0.
            train_l:
                a boolean flag denoting whether or not the lengthscale(s) should 
                be optimised along with any other hyper and/or variational parameters.
                Default: False.
            fixed_ls:
                an optional argument that contains a dictionary of feature index
                (key) lengthscale (value) pairs that are to be held fixed if ARD 
                is being used.
                Default: None.
            ard:
                a boolean flag denoting whether or not to have different lengthscales
                for different feature dimensions. This is only useful if `fixed_ls` 
                True so that different lengthscales can be learned.
                Default: False
            **mean_func_kwargs: 
                These are further keyword arguments that are passed to the prior mean
                function object. See `mean_functions.py` for more details.

                train_mean_func: 
                    a boolean flag denoting whether or not any prior mean function
                    parameters should be optimised along with any other hyper an/or
                    variational paramters. This is ignored for mean functions that
                    have no parameters (e.g. ZeroMean/'zero').
                    Default: False
                prior_mean_init: 
                    a float representing the (initial) constant value of the prior mean
                    if the mean function is 'constant'. This is ignored if the mean
                    function is something other than 'constant'.
                    Default: 0.0.
                positive_gradient_init:
                    a float representing the (initial) value of m. This is ignored
                    if the mean function is something other than 'ofs'.
                    Default: 1.0.
                offset_init:
                    a float representing the (initial) value of c. This is ignored
                    if the mean function is something other than 'ofs'.
                    Default: 2.0.
                guideprice_dim:
                    the index of the guideprice dimension. This is ignored if the 
                    mean function is not 'ofs' or 'fss'.
                    Default: None.
                sharpness_init:
                    a float representing the (initial) value of s. This is ignored
                    if the mean function is not 'fss'.
                    Default: 1.0.
                peak_init:
                    a float representing the (initial) value of p. This is ignored
                    if the mean function is not 'fss'.
                    Default: 0.0.
                loc_init:
                    a float representing the (initial) value of l. This is ignored
                    if the mean function is not 'fss'.
                    Default: 0.0.
        
        
    Example usage:
    
        >>> import gp
        
        # to initialise a sparse variational GP:
        >>> svgp = gp.models.SparseVariationalGaussianProcess(2, num_inducing=100, likelihood='Gaussian') 
        
        # to train the GP, see `training.py`.
        
        # Regression only:
            # to obtain a collection of n prior samples (note that the prior is over
            # the variable f, not y or t):
            >>> my_test_points = torch.randn((50, 2))
            >>> samps = svgp.prior(my_test_points).sample((n,))

            # to obtain (marginal) prior predictive mean and stds:
            >>> prior_dist = svgp.prior(my_test_points)
            >>> prior_means = prior_dist.mean
            >>> prior_stds = prior_dist.variance.sqrt()

            # to obtain a collection of n posterior predictive samples:
            >>> samps = svgp(my_test_points).sample((n,))

            # to obtain (marginal) posterior predictive mean and stds:
            >>> post_dist = svgp(my_test_points)
            >>> post_means = post_dist.mean
            >>> post_stds = post_dist.variance.sqrt()
        
        # Classification only:
            # to obtain prior class probabilities:
            >>> my_test_points = torch.randn((50, 2))
            >>> probs = svgp.likelihood.posterior_predictive(svgp.prior(my_test_points).probs
            
            # to obtain posterior predictive class probabilities:
            >>> probs = svgp(my_test_points).probs
        
        # to estimate the negative ELBO for a batch of the dataset via Monte Carlo integration:
        >>> neg_elbo, _ = svgp.loss(my_X_batch, my_y_batch, my_X.shape[0], num_samples=16)
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_inducing: int,
        likelihood: str,
        sigma_y: float = 1e-2,
        train_sigma_y: bool = False,
        **prior_params,
    ):
        super().__init__()
        
        # global attributes
        self.num_inputs = num_inputs
        self.num_inducing = num_inducing
        self.prior = GPPrior(num_inputs, **prior_params)
        if likelihood.lower() == 'gaussian':
            self.likelihood = GaussianLikelihood(sigma_y=sigma_y, train_sigma_y=train_sigma_y)
        elif likelihood.lower() == 'bernoulli':
            self.likelihood = BernoulliLikelihood()
        else:
            raise NotImplementedError(f"{likelihood} likelihood not recognised")
            
        # learnable parameters
        # inducing inputs Z and inducing variables u with mean vector m and covariance matrix S
        self.Z = nn.Parameter(torch.randn((num_inducing, num_inputs)), requires_grad=True)
        self.m = nn.Parameter(torch.randn((num_inducing,)), requires_grad=True)
        # parameterise the inducing variable covariance as the log diagonal and off-diagonals of the
        # Cholesky decomposition of the matrix. These values are unconstrained unlike those of 
        # the covariance matrix. This is the matrix equivalent of parameterising a positive-
        # only parameter in log-space.
        self.S_log_chol_diag = nn.Parameter(torch.log(torch.ones((num_inducing,))*0.1 + torch.randn((num_inducing,)) * 0.01), requires_grad=True)
        self.S_chol_off_diag = nn.Parameter(torch.ones((num_inducing, num_inducing))*0.001 + torch.randn((num_inducing, num_inducing)) * 0.0001, requires_grad=True)
            
    def init_inducing_variables(self, X: torch.Tensor, t: torch.Tensor):
        """
        Initialises inducing point inputs to be a random subset of the dataset for better
        training initialisation.
        """
        assert X.shape[0] >= self.num_inducing
        inducing_idx = choose_m_from_n(X.shape[0], self.num_inducing)[0]
        self.Z.data = X[inducing_idx,:]
        if isinstance(self.likelihood, BernoulliLikelihood):
            # m lives in f space rather than t space, so logistic(m) is roughly 1 or 0 at initialisation
            self.m.data = torch.where(t[inducing_idx] == 1, torch.tensor(2.0), torch.tensor(-2.0))
        elif isinstance(self.likelihood, GaussianLikelihood):
            self.m.data = t[inducing_idx,:]
    
    @property
    def S_chol(self):
        """Construct the Cholesky decomposition of the inducing variable covariance matrix
        from the nn.Parameters we have set up.
        """
        return torch.diag(self.S_log_chol_diag.exp()+1e-8) + torch.tril(self.S_chol_off_diag, diagonal=-1)
    
    @property
    def S(self):
        """Compute the inducing variable covariance matrix from its Choleksy decomposition.
        Add 1e-8 jitter for numerical stability.
        """
        return self.S_chol @ self.S_chol.T + torch.eye(self.num_inducing)*1e-8

    @property
    def q_u(self):
        """returns the Gaussian approximate posterior over the inducing variables u"""
        return torch.distributions.MultivariateNormal(self.m.squeeze(), covariance_matrix=self.S)
    
    def q_fn(self, X_batch):
        """returns the Gaussian approximate marginal posteriors over the latent
        function values corresponding to the each datapoint in the minibatch.
        
        This is a direct implementation of Hensman 2015 section 4."""
        
        K_mm = self.prior.kernel(self.Z)
        K_nm = self.prior.kernel(X_batch, self.Z)
        assert K_nm.shape[1] == self.num_inducing
        K_nn_diag = self.prior.kernel.diagonal(X_batch)
        L_mm = torch.linalg.cholesky(K_mm)    
        A = torch.cholesky_solve(K_nm.T, L_mm).T
        
        f_mu = (A @ self.m).squeeze() + self.prior.mean(X_batch)
        f_vars = K_nn_diag - torch.einsum('ij,jk,ki->i', [A, K_mm - self.S, A.T])
        # f_vars = K_nn_diag - (A @ (K_mm - self.S) @ A.T).diagonal() # more readable but slower than above line
        return torch.distributions.Normal(f_mu, f_vars.sqrt())
    
    def forward(self, X_test):
        """Computes the posterior predictive distribution.
        Returns a torch.distributions.MultivariateNormal if doing regression
        and a torch.distributions.Bernoulli if doing classification."""
        q_fn_test = self.q_fn(X_test)
        return self.likelihood.posterior_predictive(q_fn_test)
    
    def E_log_likelihood(self, X_batch, t_batch, num_samples=1):
        """Computes a Monte Carlo estimate of the expected log likelihood 
        for a minibatch of data. `num_samples` determines the number of Monte Carlo 
        samples used in the estimate.
        """
        fn_samples = self.q_fn(X_batch).rsample((num_samples,))
        preds = self.likelihood(fn_samples)
        return self.likelihood.log_prob(predictions=preds, targets=t_batch.squeeze()).mean(0).sum()
    
    def kl(self):
        """Computes the KL divergence between prior and posterior distributions over
        the inducing variables.
        """
        return torch.distributions.kl.kl_divergence(self.q_u, self.prior(self.Z)).sum()
    
    def loss(self, X_batch, t_batch, dataset_size, num_samples=1):
        """Estimates the ELBO via standard Monte Carlo variational inference. Since 
        torch optimisers do gradient *descent*, this returns as estimate of the *negative* 
        ELBO. It also returns a dictionary of useful metrics including the ELBO
        and any trainable hyperparameters. `num_samples` determines the number of 
        Monte Carlo samples used in the estimate. Higher is more accurate but costlier.
        """
        batch_E_ll = self.E_log_likelihood(X_batch, t_batch, num_samples=num_samples)
        E_ll = (dataset_size / X_batch.shape[0]) * batch_E_ll
        kl = self.kl()
        elbo = E_ll - kl
        
        metrics = {
            "elbo": elbo.detach().item(),
            "Exp ll": E_ll.detach().item(),
            "kl": kl.detach().item(),
        }
        
        for name, param in self.prior.named_parameters():
            if param.requires_grad:
                if (self.prior.kernel.ard and name == 'kernel.log_l'):
                    pass
                elif 'log' in name:
                    if 'log_' in name:
                        name = name.replace("log_", "")
                    elif '_log' in name:
                        name = name.replace("_log", "")
                    else:
                        name = name.replace("log", "")
                    metrics[name] = param.exp().detach().item()
                else:
                    metrics[name] = param.detach().item()

        for name, param in self.likelihood.named_parameters():
            if param.requires_grad:
                if 'log' in name:
                    if 'log_' in name:
                        name = name.replace("log_", "")
                    elif '_log' in name:
                        name = name.replace("_log", "")
                    else:
                        name = name.replace("log", "")
                    metrics[name] = param.exp().detach().item()
                else:
                    metrics[name] = param.detach().item()
            
        return - elbo, metrics
    
    
    

class SparseOrthogonalVariationalGaussianProcess(nn.Module):
    """Represents a Sparse Orthogonal Variational Gaussian Process.
    
    The implementation closely follows
        'Sparse Orthogonal Variational Inference for Gaussian Processes'
        Shi , Titsias, Mnih (2019)
    including variable names.

    For regression, likelihood must be Gaussian.
    For classification, likelihood must be Bernoulli.
    
    Args:
        num_inputs: 
            an integer denoting the number of input dimensions.
        num_inducing:
            an integer denoting the number of inducing points to be used in the standard
            (non-orthogonal) set. The higher the better, but the more computationally 
            expensive. 
        num_orthogonal_inducing:
            an integer denoting the number of inducing points to be used in the 
            orthogonal set. The higher the better, but the more computationally 
            expensive. The total number of inducing points is given by:
                total_inducing = num_inducing + num_orthogonal_inducing.
            More points can be used in this model than in the 
            SparseVariationalGaussianProcess for the same computational cost due to the
            orthogonal decomposition of the Gaussian process.
        likelihood:
            a string denoting the choice of likelihood function. Options are:
                'Gaussian': to be used for regression,
                'Bernoulli': to be used for binary classification. 
            Note that character case is ignored.
        sigma_y: 
            a positive float denoting the observation noise/std of the Gaussian
            likelihood. This is ignored unless the likelihood is 'Gaussian'.
            Default: 1e-2.
        train_sigma_y:
            a boolean flag denoting whether or not sigma_y should be optimised 
            along with any other hyper and/or variational parameters. This is 
            ignored unless the likelihood is 'Gaussian'.
            Default: False.
        **prior_params:
            These are further keyword arguments that are passed to the GP prior
            object. See `GPPrior` for more details.
                
            covariance_function:
                a string denoting the choice of covariance function. Options are:
                    'exponential',
                    'matern-1.5',
                    'matern-2.5',
                    'squared-exponential'.
                Default: 'squared-exponential'.
            mean_function:
                a string denoting the choice of prior mean function. Options are:
                    'zero',
                    'constant',
                    'ofs',
                    'fss'.
                Default: 'zero'.
            l: 
                a positive float representing the lengthscale hyperparameter of the 
                covariance function. If ARD is being used, this is the (initial) 
                lengthscale for every dimension, unless any fixed lengthscales are 
                specified via `fixed_ls`.
                Default: 1.0.
            train_l:
                a boolean flag denoting whether or not the lengthscale(s) should 
                be optimised along with any other hyper and/or variational parameters.
                Default: False.
            fixed_ls:
                an optional argument that contains a dictionary of feature index
                (key) lengthscale (value) pairs that are to be held fixed if ARD 
                is being used.
                Default: None.
            ard:
                a boolean flag denoting whether or not to have different lengthscales
                for different feature dimensions. This is only useful if `fixed_ls` 
                True so that different lengthscales can be learned.
                Default: False
            **mean_func_kwargs: 
                These are further keyword arguments that are passed to the prior mean
                function object. See `mean_functions.py` for more details.

                train_mean_func: 
                    a boolean flag denoting whether or not any prior mean function
                    parameters should be optimised along with any other hyper an/or
                    variational paramters. This is ignored for mean functions that
                    have no parameters (e.g. ZeroMean/'zero').
                    Default: False
                prior_mean_init: 
                    a float representing the (initial) constant value of the prior mean
                    if the mean function is 'constant'. This is ignored if the mean
                    function is something other than 'constant'.
                    Default: 0.0.
                positive_gradient_init:
                    a float representing the (initial) value of m. This is ignored
                    if the mean function is something other than 'ofs'.
                    Default: 1.0.
                offset_init:
                    a float representing the (initial) value of c. This is ignored
                    if the mean function is something other than 'ofs'.
                    Default: 2.0.
                guideprice_dim:
                    the index of the guideprice dimension. This is ignored if the 
                    mean function is not 'ofs' or 'fss'.
                    Default: None.
                sharpness_init:
                    a float representing the (initial) value of s. This is ignored
                    if the mean function is not 'fss'.
                    Default: 1.0.
                peak_init:
                    a float representing the (initial) value of p. This is ignored
                    if the mean function is not 'fss'.
                    Default: 0.0.
                loc_init:
                    a float representing the (initial) value of l. This is ignored
                    if the mean function is not 'fss'.
                    Default: 0.0.
        
        
    Example usage:
    
        >>> import gp
        
        # to initialise a sparse variational GP:
        >>> sovgp = gp.models.SparseOrthogonalVariationalGaussianProcess(2, num_inducing=100, num_orthogonal_inducing=100, likelihood='Gaussian') 
        
        # to train the GP, see `training.py`.
        
        # Regression only:
            # to obtain a collection of n prior samples (note that the prior is over
            # the variable f, not y or t):
            >>> my_test_points = torch.randn((50, 2))
            >>> samps = sovgp.prior(my_test_points).sample((n,))

            # to obtain (marginal) prior predictive mean and stds:
            >>> prior_dist = sovgp.prior(my_test_points)
            >>> prior_means = prior_dist.mean
            >>> prior_stds = prior_dist.variance.sqrt()

            # to obtain a collection of n posterior predictive samples:
            >>> samps = sovgp(my_test_points).sample((n,))

            # to obtain (marginal) posterior predictive mean and stds:
            >>> post_dist = sovgp(my_test_points)
            >>> post_means = post_dist.mean
            >>> post_stds = post_dist.variance.sqrt()
        
        # Classification only: (i.e. likelihood='Bernoulli')
            # to obtain prior class probabilities:
            >>> my_test_points = torch.randn((50, 2))
            >>> probs = sovgp.likelihood.posterior_predictive(sovgp.prior(my_test_points).probs
            
            # to obtain posterior predictive class probabilities:
            >>> probs = sovgp(my_test_points).probs
        
        # to estimate the negative ELBO for a batch of the dataset via Monte Carlo integration:
        >>> neg_elbo, _ = sovgp.loss(my_X_batch, my_y_batch, my_X.shape[0], num_samples=16)
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_inducing: int,
        num_orthogonal_inducing: int,
        likelihood: str,
        sigma_y: float = 1e-2,
        train_sigma_y: bool = False,
        **prior_params,
    ):
        super().__init__()
        
        # global attributes
        self.num_inputs = num_inputs
        self.num_inducing = num_inducing
        self.num_orthogonal_inducing = num_orthogonal_inducing
        self.prior = GPPrior(num_inputs, **prior_params)
        if likelihood.lower() == 'gaussian':
            self.likelihood = GaussianLikelihood(sigma_y=sigma_y, train_sigma_y=train_sigma_y)
        elif likelihood.lower() == 'bernoulli':
            self.likelihood = BernoulliLikelihood()
        else:
            raise NotImplementedError(f"{likelihood} likelihood not recognised")
            
            
        # variational parameters:
        
        # inducing points/inputs
        self.Z = nn.Parameter(torch.randn((num_inducing, num_inputs)), requires_grad=True)
        self.O = nn.Parameter(torch.randn((num_orthogonal_inducing, num_inputs)), requires_grad=True)
        # inducing output means
        self.m_u = nn.Parameter(torch.randn((num_inducing,)), requires_grad=True)
        self.m_v = nn.Parameter(torch.randn((num_orthogonal_inducing,)), requires_grad=True)
        # inducing output covariance function cholesky decomposition parameterisations
        self.Lu_log_diag = nn.Parameter(torch.log(torch.ones((num_inducing,))*0.1 + torch.randn((num_inducing,)) * 0.01), requires_grad=True)
        self.Lu_off_diag = nn.Parameter(torch.randn((num_inducing, num_inducing)) * 0.001, requires_grad=True)
        self.Lv_log_diag = nn.Parameter(torch.log(torch.ones((num_orthogonal_inducing,))*0.1 + torch.randn((num_orthogonal_inducing,)) * 0.01), requires_grad=True)
        self.Lv_off_diag = nn.Parameter(torch.randn((num_orthogonal_inducing, num_orthogonal_inducing)) * 0.001, requires_grad=True)
            
    def init_inducing_variables(self, X: torch.Tensor, t: torch.Tensor):
        """
        Initialises inducing point inputs to be a random subset of the dataset for better
        training initialisation.
        """
        assert X.shape[0] >= self.num_inducing
        inducing_idx = choose_m_from_n(X.shape[0], self.num_inducing)[0]
        orth_inducing_idx = choose_m_from_n(X.shape[0], self.num_orthogonal_inducing)[0]
        self.Z.data = X[inducing_idx,:]
        self.O.data = X[orth_inducing_idx,:]
        if isinstance(self.likelihood, BernoulliLikelihood):
            # m lives in f space rather than t space, so logistic(m) is roughly 1 or 0 at initialisation
            self.m_u.data = torch.where(t[inducing_idx] == 1, torch.tensor(2.0), torch.tensor(-2.0))
            self.m_v.data = torch.where(t[orth_inducing_idx] == 1, torch.tensor(2.0), torch.tensor(-2.0))
        elif isinstance(self.likelihood, GaussianLikelihood):
            self.m_u.data = t[inducing_idx,:]
            self.m_v.data = t[orth_inducing_idx,:]
    
    @property
    def Lu(self):
        """Construct the Cholesky decomposition of the inducing variable covariance matrix
        from the nn.Parameters we have set up. This is for the standard set.
        """
        return torch.diag(self.Lu_log_diag.exp()+1e-8) + torch.tril(self.Lu_off_diag, diagonal=-1)

    @property
    def Lv(self):
        """Construct the Cholesky decomposition of the inducing variable covariance matrix
        from the nn.Parameters we have set up. This is for the orthogonal set.
        """
        return torch.diag(self.Lv_log_diag.exp()+1e-8) + torch.tril(self.Lv_off_diag, diagonal=-1)
    
    @property
    def Su(self):
        """Compute the inducing variable covariance matrix from its Choleksy decomposition.
        Add 1e-8 jitter for numerical stability. This is for the standard set.
        """
        return self.Lu @ self.Lu.T + torch.eye(self.num_inducing)*1e-8
    
    @property
    def Sv(self):
        """Compute the inducing variable covariance matrix from its Choleksy decomposition.
        Add 1e-8 jitter for numerical stability. This is for the standard set.
        """
        return self.Lv @ self.Lv.T + torch.eye(self.num_orthogonal_inducing)*1e-8

    @property
    def q_u(self):
        """returns the Gaussian approximate posterior over the inducing variables u"""
        return torch.distributions.MultivariateNormal(self.m_u.squeeze(), covariance_matrix=self.Su)
    
    @property
    def q_v(self):
        """returns the Gaussian approximate posterior over the orthogonal set of 
        inducing variables v_{\perp}. See Shi et al. section 3.3 top rhs of the page.
        This is the first probability distribution in the second KL term in equation 8
        of the paper."""
        return torch.distributions.MultivariateNormal(self.m_v.squeeze(), covariance_matrix=self.Sv)
    
    @property
    def Kuu(self):
        return self.prior.kernel(self.Z)
    
    @property
    def Kvv(self):
        return self.prior.kernel(self.O)      
        
    
    def q_fn(self, X_batch):
        """returns the Gaussian approximate marginal posteriors over the latent
        function values corresponding to the each datapoint in the minibatch.
        
        This is a direct implementation of Shi et al. 2019 Appendix D.1
        Algorithm 1 up to the end of line 7."""
        
        Lu0 = torch.linalg.cholesky(self.Kuu + torch.eye(self.num_inducing)*1e-8)
        # Lu0_inv = torch.linalg.solve_triangular(Lu0, torch.eye(self.num_inducing))
        Kuv = self.prior.kernel(self.Z, self.O)
        A = torch.linalg.solve_triangular(Lu0, Kuv, upper=False)
        Cvv = self.Kvv - A.T @ A
        Lv0 = torch.linalg.cholesky(Cvv + torch.eye(self.num_orthogonal_inducing)*1e-8)
        # Lv0_inv = torch.linalg.solve_triangular(Lv0, torch.eye(self.num_orthogonal_inducing))
        Kuf = self.prior.kernel(self.Z, X_batch)
        Kvf = self.prior.kernel(self.O, X_batch)
        B = torch.linalg.solve_triangular(Lu0, Kuf, upper=False)
        Cvf = Kvf - A.T @ B
        D = torch.linalg.solve_triangular(Lv0, Cvf, upper=False)
        E = torch.linalg.solve_triangular(Lu0.T, B, upper=False)
        F = self.Lu.T @ E
        G = torch.linalg.solve_triangular(Lv0, D, upper=False)
        H = self.Lv.T @ G
        
        f_mu = E.T @ self.m_u + G.T @ self.m_v + self.prior.mean(X_batch)
        f_vars = self.prior.kernel.diagonal(X_batch) + (F * F).sum(0) - (B * B).sum(0) + (H * H).sum(0) - (D * D).sum(0)
        
        return torch.distributions.Normal(f_mu, f_vars.sqrt())
    
    def forward(self, X_test):
        """Computes the posterior predictive distribution.
        Returns a torch.distributions.MultivariateNormal if doing regression
        and a torch.distributions.Bernoulli if doing classification."""
        q_fn_test = self.q_fn(X_test)
        return self.likelihood.posterior_predictive(q_fn_test)
    
    def E_log_likelihood(self, X_batch, t_batch, num_samples=1):
        """Computes a Monte Carlo estimate of the expected log likelihood 
        for a minibatch of data. `num_samples` determines the number of Monte Carlo 
        samples used in the estimate.
        """
        fn_samples = self.q_fn(X_batch).rsample((num_samples,))
        preds = self.likelihood(fn_samples)
        return self.likelihood.log_prob(predictions=preds, targets=t_batch.squeeze()).mean(0).sum()
    
    def kl_u(self):
        """Computes the KL divergence between prior and posterior distributions over
        the standard inducing variables u.
        """
        return torch.distributions.kl.kl_divergence(self.q_u, self.prior(self.Z)).sum()
    
    def kl_v(self):
        """Computes the KL divergence between prior and posterior distributions over
        the orthogonal inducing variables v_{\perp}."""
        # this is repeated computation from self.q_fn(). The code can be improved by 
        # using Cvv computed there directly here rather than recomputing it. Note
        # that Cvv := Kvv - A.t @ A .
        Lu0 = torch.linalg.cholesky(self.Kuu + torch.eye(self.num_inducing)*1e-8)
        Kuv = self.prior.kernel(self.Z, self.O)
        A = torch.linalg.solve_triangular(Lu0, Kuv, upper=False)
        p_perp_v = torch.distributions.MultivariateNormal(torch.zeros((self.num_orthogonal_inducing,)), self.Kvv - A.T @ A)  
        return torch.distributions.kl.kl_divergence(self.q_v, p_perp_v).sum()
    
    def loss(self, X_batch, t_batch, dataset_size, num_samples=1):
        """Estimates the ELBO via standard Monte Carlo variational inference. Since 
        torch optimisers do gradient *descent*, this returns as estimate of the *negative* 
        ELBO. It also returns a dictionary of useful metrics including the ELBO
        and any trainable hyperparameters. `num_samples` determines the number of 
        Monte Carlo samples used in the estimate. Higher is more accurate but costlier.
        """
        batch_E_ll = self.E_log_likelihood(X_batch, t_batch, num_samples=num_samples)
        E_ll = (dataset_size / X_batch.shape[0]) * batch_E_ll
        kl_u = self.kl_u()
        kl_v = self.kl_v()
        elbo = E_ll - kl_u - kl_v
        
        metrics = {
            "elbo": elbo.detach().item(),
            "Exp ll": E_ll.detach().item(),
            "kl_u": kl_u.detach().item(),
            "kl_v": kl_v.detach().item(),
        }
        
        for name, param in self.prior.named_parameters():
            if param.requires_grad:
                if (self.prior.kernel.ard and name == 'kernel.log_l'):
                    pass
                elif 'log' in name:
                    if 'log_' in name:
                        name = name.replace("log_", "")
                    elif '_log' in name:
                        name = name.replace("_log", "")
                    else:
                        name = name.replace("log", "")
                    metrics[name] = param.exp().detach().item()
                else:
                    metrics[name] = param.detach().item()

        for name, param in self.likelihood.named_parameters():
            if param.requires_grad:
                if 'log' in name:
                    if 'log_' in name:
                        name = name.replace("log_", "")
                    elif '_log' in name:
                        name = name.replace("_log", "")
                    else:
                        name = name.replace("log", "")
                    metrics[name] = param.exp().detach().item()
                else:
                    metrics[name] = param.detach().item()
            
        return - elbo, metrics
        