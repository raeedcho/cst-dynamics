import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.neighbors import KDTree

def estimate_neural_tangling(data,x,dx,num_neighbors=None,num_sample_points=None,const_norm=True,take_max=False):
    '''
    Estimate tangling of neural activity using a subset of the trials in the data.

    Args:
        data(pd.DataFrame): dataframe with trial_id and columns for x and dx
        x(str): column name for x
        dx(str): column name for dx
        num_neighbors(int): number of nearest neighbors for each point to use in tangling calculation
        num_sample_points(int): number of randomly selected timepoints to use in tangling calculation
        const_norm(bool): if True, use a constant normalization term to prevent divide by zero errors
            If false, use a normalization term that is proportional to the variance of the data
        take_max(bool): if True, use the maximum tangling value for each point, otherwise use the 99th percentile

    Returns:
        (np.ndarray): tangling values for each point
    '''
    assert type(x)==str and type(dx)==str, 'x and dx must be strings to specify column names in data'

    norm_adjust = 1e-6 if const_norm else 0.1*sum(x.var())
    
    # if num_trials is None:
    #     # subselect data to use in tangling calculation
    #     data
    # else:
    #     data_sub = data

    if num_neighbors is None and num_sample_points is None:
        full_tang = squareform(pdist(data[dx],metric='sqeuclidean')/(pdist(data[x],metric='sqeuclidean')+norm_adjust))
    elif num_neighbors is None and num_sample_points is not None:
        data_sample = data.sample(n=num_sample_points)
        full_tang = cdist(data[dx],data_sample[dx],metric='sqeuclidean')/(cdist(data[x],data_sample[x],metric='sqeuclidean')+norm_adjust)
    elif num_neighbors is not None and num_sample_points is None:
        # use nearest neighbors to calculate tangling by using a KD tree
        # do I want to create my own distance metric based on tangling?...
        # build a KD tree from the data
        kd_tree = KDTree(data[dx])
        # find the nearest neighbors for each point
        nearest_neighbors = kd_tree.query(data[dx],k=num_neighbors,return_distance=False)
        # calculate tangling for each point
        full_tang = cdist(data[dx],data_sample[dx],metric='sqeuclidean')/(cdist(data[x],data_sample[x],metric='sqeuclidean')+norm_adjust)
        raise NotImplementedError
    else:
        raise NotImplementedError

    q = full_tang.max(axis=1) if take_max else np.percentile(full_tang,99,axis=1)

    return q

def rand_sample_tangling(num_samples,**tangling_kwargs):
    '''
    Wrapper around estimate_neural_tangling to sample tangling values for a number of random subsets of the data.
    '''
    return pd.concat(
        [
            pd.Series(
                estimate_neural_tangling(**tangling_kwargs),
                index=df.index,
            ) for _ in range(num_samples)
        ],
        axis=0,
    )
