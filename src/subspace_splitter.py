'''
This module implements Brian's subspace splitting code from MATLAB in Python,
using PyManopt. 
'''
import autograd.numpy as anp
import pymanopt

def split_subspace(cond1_cov,cond2_cov,cutoff='auto',plot=False):
    '''
    This is the main function to split a subspace into shared and unique
    subspaces for two conditions.

    Parameters
    ----------
    cond1_cov : array_like
        The covariance matrix for condition 1 OR data matrix (samples x dimensions) from condition 1
    cond2_cov : array_like
        The covariance matrix for condition 2 OR data matrix (samples x dimensions) from condition 2
    cutoff : float or 'auto'
        The cutoff parameter changes how the algorithm will
        determine the dimensionality of the unique spaces.
        There are three possibilites: 

        - scalar [0, 1): variance explained cutoff.
        e.g. cutoff=0.05 means that the unique space
        calculated for condition 1 will contain, at most, 5%
        of the total variance during condition 2. 

        - scalar [1 D]: number of total unique dimensions. 
        e.g. cutoff=1 will attempt to find only a single
        unique dimension.

        - 'auto' (default): uses a heuristic to determine the
        best split of unique/shared dimensions.
    plot : bool
        Whether or not to plot

    Returns
    -------
    Q : dict
    varexp : dict
    Qs : dict
    '''
    pass