import numpy as np
from scipy.linalg import null_space
import torch
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

def fit_dekodec(X_conds, var_cutoff=0.99, do_plot=True, combinations=None):
    """
    Splits data into orthogonal subspaces containing condition-unique and 
    condition-shared activity.

    Parameters
    ----------
    X_conds : dict of numpy arrays
        dict of matrices containing neural firing rates for different conditions,labelled by condition.
        In each array, rows correspond to samples and columns correspond to neural dimensions.
        All arrays must have the same dimensionality.
    var_cutoff : float, optional
        Fraction variance cutoff used to delineate potent and null spaces.
        If no cutoff is given, the default value is 0.99, i.e. the potent space
        will explain a minimum of 99% of the total variance.
    do_plot : bool, optional
        Flag to return plot (True) [default] or not (False)
    combinations : list of tuples, optional
        Custom list of combinations to check. Currently not converted from original MATLAB code.

    Returns
    -------
    subspaces : dict
        Dictionary containing 'unique' and 'shared' fields, which contain the axes for each identified subspace.
        Together, they form a full orthonormal basis.
    
    Created by Brian Dekleva, 2023-05-02
    Adapted for Python by Raeed Chowdhury, 2023-07-19
    """
    
    num_conds = len(X_conds)

    assert num_conds > 1, 'Must have at least two conditions to compare'
    assert combinations is None, 'Custom combinations not yet implemented'

    cond_unique_projmats = {
        cond: get_cond_unique_basis(X_conds,cond,var_cutoff=var_cutoff)
        for cond in X_conds
    }
    subspaces = orthogonalize_unique_spaces(X_conds,cond_unique_projmats)
    subspaces['shared'] = max_var_rotate(
        null_space(np.column_stack(tuple(subspaces.values()))),
        np.row_stack(tuple(X_conds.values())),
    )

    return subspaces

def get_cond_unique_basis(X_conds,which_cond,var_cutoff=0.99):
    """
    Calculates the conditional unique basis for a matrix based on the percent variance cutoff.

    Parameters:
    -----------
    X_conds: dict of numpy arrays
        List of matrices containing neural firing rates for different conditions.
        In each array, rows correspond to samples and columns correspond to neural dimensions.
        All arrays must have the same dimensionality.
    which_cond : dict key
        condition to calculate the conditional unique basis for.
    var_cutoff : float, optional (default=0.99)
        The fraction variance cutoff for the eigenvalues.

    Returns:
    --------
    cond_unique_projmat : array-like, shape (n_features, n_unique_dims)
        The condition unique projection matrix.
    """

    assert len({X.shape[1] for X in X_conds.values()}) == 1, 'All conditions must have the same number of dimensions'

    X_cond = X_conds[which_cond]
    X_notcond = np.row_stack([
        X for cond,X in X_conds.items()
        if cond != which_cond
    ])

    _,notcond_null = get_potent_null(X_notcond, var_cutoff=var_cutoff)
    cond_unique_projmat = max_var_rotate(notcond_null,X_cond)
    num_unique_dims = get_num_projected_dims_to_keep(
        X_cond,
        cond_unique_projmat,
        var_cutoff=var_cutoff
    )

    return cond_unique_projmat[:,:num_unique_dims]

def orthogonalize_unique_spaces(X_conds,cond_unique_projmats):
    """
    Orthogonalize the unique subspaces of the input data matrices.

    Parameters
    ----------
    X_conds : dict
        A dictionary of input data matrices, where each key is a condition label
        and each value is a matrix with shape (n_samples, n_features).
    cond_unique_projmats : dict
        A dictionary of projection matrices, where each key is a condition label
        and each value is a matrix with shape (n_features, n_unique_dims).

    Returns
    -------
    dict
        A dictionary of orthogonalized projection matrices, where each key is a condition label and each value is a matrix with shape (n_features, n_unique_dims).

    """
    num_unique_dims = [projmat.shape[1] for projmat in cond_unique_projmats.values()]
    total_unique_dims = np.sum(num_unique_dims)
    if total_unique_dims == 0:
        raise ValueError('No unique dimensions found')

    Z = torch.from_numpy(np.row_stack(tuple(X_conds.values())))
    Z_uniques = torch.from_numpy(np.column_stack([
        Z @ torch.from_numpy(projmat)
        for projmat in cond_unique_projmats.values()
    ]))

    manifold = pymanopt.manifolds.Stiefel(Z.shape[1],total_unique_dims)

    @pymanopt.function.pytorch(manifold)
    def cost(Q):
        return torch.sum(torch.square(Z @ Q - Z_uniques))

    problem = pymanopt.Problem(manifold,cost)
    optimizer = pymanopt.optimizers.TrustRegions()
    result = optimizer.run(problem)
    
    Q_all_uniques = flip_positive(result.point)

    Q_unique = {
        cond: arr for cond,arr in zip(
            X_conds.keys(),
            np.split(Q_all_uniques,np.cumsum(num_unique_dims),axis=1)[:-1]
        )
    }

    return Q_unique

def get_potent_null(X, var_cutoff=0.99):
    """
    Calculates the potent and null spaces for a matrix based on the percent variance cutoff.
    
    Parameters
    ----------
    X : numpy array
        Matrix containing neural firing rates.
        Rows correspond to samples and columns correspond to neural dimensions.
    var_cutoff : float, optional
        Percent variance cutoff used to delineate potent and null spaces.
        If no cutoff is given, the default value is 0.99, i.e. the potent space
        will explain a minimum of 99% of the total variance.9.

    Returns
    -------
    potent_projmat : numpy array
        Projection matrix to transform data into the potent space basis
    null_projmat : numpy array
        Projection matrix to transform data into the null space basis
    """

    X_centered = X - np.mean(X, axis=0)

    num_dims,_ = get_dimensionality(X_centered, var_cutoff=var_cutoff)
    _, _, Vh = np.linalg.svd(X_centered, full_matrices=False)

    potent_projmat = Vh[:num_dims,:].T
    null_projmat = Vh[num_dims:,:].T

    return potent_projmat, null_projmat

def get_dimensionality(X, var_cutoff=0.99):
    """
    Calculates the dimensionality of a matrix based on the percent variance cutoff.

    Parameters
    ----------
    X : numpy array
        Matrix containing neural firing rates.
        Rows correspond to samples and columns correspond to neural dimensions.
    var_cutoff : float, optional
        Percent variance cutoff used to delineate potent and null spaces.
        If no cutoff is given, the default value is 0.99, i.e. the potent space
        will explain a minimum of 99% of the total variance.

    Returns
    -------
    num_dims : int
        Number of dimensions in the matrix.
    eigenvalues : numpy array
        Eigenvalues of the matrix.
    """

    assert 0 < var_cutoff < 1, 'Variance cutoff must be between 0 and 1'

    X_centered = X - np.mean(X, axis=0)
    _, S, _ = np.linalg.svd(X_centered, full_matrices=False)

    eigenvalues = S**2
    cumulative_variance_explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_dims = np.sum(cumulative_variance_explained <= var_cutoff) + 1

    return num_dims, eigenvalues

def max_var_rotate(proj_mat,X):
    """
    Rotate the columns of the projection matrix proj_mat to maximize the variance
    of the data (X) projected through it.

    Parameters
    ----------
    proj_mat : numpy.ndarray
        The initial projection matrix with shape (n_features, n_components).
    X : numpy.ndarray
        The input data matrix with shape (n_samples, n_features).

    Returns
    -------
    numpy.ndarray
        The updated projection matrix with shape (n_features, n_components).
    """
    X_centered = X - np.mean(X,axis=0)

    projected_activity = X_centered @ proj_mat
    _,_,Vh = np.linalg.svd(projected_activity,full_matrices=False)
    new_proj_mat = proj_mat @ Vh.T

    return new_proj_mat

def get_num_projected_dims_to_keep(X,proj_mat,var_cutoff=0.99):
    """
    Get number of dimensions to keep after projecting data (X) through proj_mat,
    based on the variance explained by the original data. Only keep dimensions that
    explain at least as much variance as the last original PC.

    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix with shape (n_samples, n_features).
    proj_mat : numpy.ndarray
        The projection matrix with shape (n_features, n_components).
    var_cutoff : float, optional
        The proportion of variance to determine original data dimensionality (default is 0.99).

    Returns
    -------
    int
        The number of dimensions to keep after projection.

    """
    assert 0 < var_cutoff < 1, 'Variance cutoff must be between 0 and 1'

    full_dimensionality,full_eigvals = get_dimensionality(X, var_cutoff=var_cutoff)
    _,proj_eigvals = get_dimensionality(X @ proj_mat, var_cutoff=var_cutoff)
    proj_var_explained = proj_eigvals / np.sum(full_eigvals)
    proj_dim_var_cutoff = full_eigvals[full_dimensionality-1] / np.sum(full_eigvals)
    num_proj_dims = np.sum(proj_var_explained >= proj_dim_var_cutoff)
    
    return num_proj_dims


def flip_positive(Q):
    return Q