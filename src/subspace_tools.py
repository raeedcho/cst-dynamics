# a set of tools to manipulate and analyze geometry of high dimensional data

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
import scipy.linalg as la

from . import util

def find_joint_subspace(df,signal,condition='task',num_dims=15,remove_mean=False,orthogonalize=True):
    '''
    Find a joint subspace given multiple datasets in the same full-D space 
    and a number of dimensions to use for each dataset.

    This function will first reduce the dimensionality of the datasets to num_dims using PCA,
    then concatenate the resulting PCs to form the joint subspace. Lastly, it will orthogonalize
    the joint subspace using SVD. The result will be a projection matrix from full-D space to
    the joint subspace (n_features x (num_dims*num_conditions)).

    Note: This function will not remove the mean of the datasets before performing PCA by default.

    Arguments:
        df - (pd.DataFrame) DataFrame containing data (e.g. firing rates) and condition (e.g. task).
            Data will be grouped by the provided condition column to form multiple datasets.
            Each element of df[signal] is a numpy array with features along columns
            and optionally observations along rows. These arrays will be stacked via
            np.row_stack() to form a single data matrix for each dataset.
        signal - (str) name of column in df containing data
        condition - (str) name of column in df containing condition labels
        num_dims - (int) number of dimensions to use for each data matrix to compose
            the joint subspace.
        remove_mean - (bool) whether or not to remove the mean of X and Y before
            performing PCA. Default is False
        orthogonalize - (bool) whether or not to orthogonalize the joint subspace
            using SVD. Default is True.

    Returns:
        (numpy array) projection matrix from full-D space to joint subspace
            (n_features x (num_dims*num_conditions))
    '''
    def get_pcs(df):
        if remove_mean:
            dim_reduction_model = PCA(n_components=num_dims)
        else:
            dim_reduction_model = TruncatedSVD(n_components=num_dims)

        dim_reduction_model.fit(np.row_stack(df[signal]))
        return dim_reduction_model.components_

    separate_pcs = (
        df
        .groupby(condition)
        .apply(get_pcs)
    )
    
    if orthogonalize:
        _,_,vt = np.linalg.svd(np.row_stack(separate_pcs),full_matrices=False)
    else:
        vt = np.row_stack(separate_pcs)

    return vt.T

def subspace_overlap_index(X,Y,num_dims=10):
    '''
    Calculate the subspace overlap index (from Elsayed et al. 2016)
    between two data matrices (X and Y), given a number of dimensions

    Arguments:
        X,Y - (numpy arrays) matrix containing data (e.g. firing rates)
            with features along columns and observations along rows
        num_dims - (int) number of dimensions to use in the subspace
            overlap calculation

    Returns:
        (float) subspace overlap index
    '''
    assert X.shape[1] == Y.shape[1], 'X and Y must have same number of features'
    assert num_dims <= X.shape[1], 'num_dims must be less than or equal to number of features in X'
    assert X.ndim == 2, 'X must be a 2D array'
    assert Y.ndim == 2, 'Y must be a 2D array'

    # mean subtract X and Y
    X_hat = X - X.mean(axis=0)
    Y_hat = Y - Y.mean(axis=0)

    # get PCA of X and Y
    pca_X = PCA(n_components=num_dims)
    pca_Y = PCA(n_components=num_dims)
    pca_X.fit(X_hat)
    pca_Y.fit(Y_hat)

    # calculate overlap
    X_var = np.sum(pca_X.explained_variance_)
    X_cov = np.cov(X_hat.T)
    Y_axes = pca_Y.components_
    soi = np.trace(Y_axes @ X_cov @ Y_axes.T)/X_var

    return soi

def bootstrap_subspace_overlap(td_grouped,signal='M1_rates',num_bootstraps=100,num_dims=10):
    '''
    Compute subspace overlap for each pair of tasks and epochs,
    with bootstrapping to get distributions

    Arguments:
        td_grouped: (pandas.GroupBy object) trial data grouped by some key (e.g. task, epoch)
        num_bootstraps: (int) number of bootstraps to perform

    Returns:
        pandas.DataFrame: dataframe with rows corresponding to each bootstrap
            of subspace overlap computed for pairs of group keys
    '''
    td_boots = []
    for boot_id in range(num_bootstraps):
        data_td = td_grouped.agg(**{
            signal: (signal,lambda rates : np.row_stack(rates.sample(frac=1,replace=True)))
        })
        proj_td = td_grouped.agg(**{
            signal: (signal,lambda rates : np.row_stack(rates.sample(frac=1,replace=True)))
        })
        td_pairs = data_td.join(
            proj_td,
            how='cross',
            lsuffix='_data',
            rsuffix='_proj',
        )

        td_pairs['boot_id'] = boot_id
        td_boots.append(td_pairs)
    
    td_boots = pd.concat(td_boots).reset_index(drop=True)
    
    td_boots['subspace_overlap'] = [
        subspace_overlap_index(data,proj,num_dims=num_dims)
        for data,proj in zip(td_boots[f'{signal}_data'],td_boots[f'{signal}_proj'])
    ]

    td_boots['subspace_overlap_rand'] = [
        subspace_overlap_index(data,util.random_array_like(data),num_dims=num_dims)
        for data,proj in zip(td_boots[f'{signal}_data'],td_boots[f'{signal}_proj'])
    ]

    return td_boots

def calc_projected_variance(X,proj_matrix):
    '''
    Calculate the variance of the data projected onto the basis set
    defined by proj_matrix
    
    Arguments:
        X - (numpy array) data to project
        proj_matrix - (numpy array) basis set to project onto
        
    Returns:
        (float) projected variance
    '''
    pass

def orth_combine_subspaces(space_list):
    '''
    Combine subspaces with in space_list to create an orthogonal basis set
    spanning all spaces

    Arguments:
        space_list - (list) list of subspaces to combine (each subspace is a
            numpy array with each column as a basis vector in neural space)

    Returns:
        np.array - orthogonal basis set spanning all subspaces in space_list
    '''
    if type(space_list) != list:
        space_list = [space_list]

    for space in space_list:
        assert type(space) == np.ndarray, 'space must be a numpy array'
        assert space.ndim == 2, 'space must be a 2D array'

    # concatenate basis vectors from each subspace
    # Then run SVD on the combined basis set to get orthogonal basis set

    return np.nan

def find_potent_null_space(X,Y):
    '''
    Runs a linear regression from X to Y to find the potent and null spaces
    of the transformation that takes X to Y. For example, X could be neural
    activity and Y could be behavioral data, and potent space would be the
    neural space that correlates best with behavior.

    Arguments:
        X (numpy array) - ndarray of input data with features along columns
            and observations along rows
        Y (numpy array) - ndarray of output data with features along columns
            and observations along rows

    Returns:
        (tuple) - tuple containing:
            potent_space - (numpy array) potent space of transformation
                shape is (X.shape[1],<potent dimensionality>)
            null_space - (numpy array) null space of transformation
                shape is (X.shape[1],<null dimensionality>)
    '''
    model = LinearRegression(fit_intercept=False)
    model.fit(X,Y)

    # model.coef_ is the coefficients of the linear regression
    # (shape: (n_targets, n_features))
    
    # null space is the set of vectors such that model.coef_ @ null_vector = 0
    # null basis is of shape (n_features, null_dim)
    null_space = la.null_space(model.coef_)
    
    # potent space is the orthogonal complement of the null space
    # which is equivalent to the row space of the model.coef_ matrix
    potent_space = la.orth(model.coef_.T)

    return potent_space,null_space