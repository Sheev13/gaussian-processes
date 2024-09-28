# Gaussian Processes

### Models
This repository contains custom implementations of various Gaussian process models for machine learning. The Gaussian process models implemented are as follows:
1. `GaussianProcess`. An exact GP used for regression with a Gaussian likelihood.
2. `SparseVariationalGaussianProcess`. An implementation of a sparse variational GP from Hensman et al. 2015. This can be used for regression (Gaussian likelihood) or classification (Bernoulli likelihood) with millions of datapoints.
3. `SparseOrthogonalVariationalGaussianProcess`. An implementation of a sparse orthogonal variational GP from Shi et al. 2019. This can be used for regression (Gaussian likelihood) or classification (Bernoulli likelihood) with millions of datapoints. This model is closer to SOTA for GP scalability than the `SparseVariationalGaussianProcess`.


### Covariance Functions
The repository contains implementations of the common members of the Matern family of covariance functions:
1. Exponential (Matern 0.5)
2. Matern 1.5
3. Matern 2.5
4. Squared-exponential (Matern $\infty$)
   

### Documentation
Each .py file is is heavily documented, and so further details on any parts of the implementation can be found therein.
