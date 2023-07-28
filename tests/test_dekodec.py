from src.dekodec import *

import numpy as np
import pymanopt

def test_fit_dekodec():
    pass
    # # Test that fit_dekodec returns a dictionary with the expected keys
    # X_conds = {
    #     'cond1': np.random.rand(10, 5),
    #     'cond2': np.random.rand(10, 5)
    # }
    # subspaces = fit_dekodec(X_conds)
    # assert isinstance(subspaces, dict)
    # assert 'unique' in subspaces
    # assert 'shared' in subspaces

    # # Test that fit_dekodec raises an error if given only one condition
    # X_conds = {'cond1': np.random.rand(10, 5)}
    # try:
    #     subspaces = fit_dekodec(X_conds)
    # except AssertionError:
    #     pass
    # else:
    #     raise AssertionError('fit_dekodec did not raise an error for one condition')

def test_get_potent_null():
    num_samples = 100
    num_features = 5
    num_true_dims = 3
    
    manifold = pymanopt.manifolds.FixedRankEmbedded(num_samples,num_features,num_true_dims)
    rand_point = manifold.random_point()
    X = rand_point.u @ np.diag(rand_point.s) @ rand_point.vt

    potent_projmat,null_projmat = get_potent_null(X)

def test_get_dimensionality():
    # test that get_dimensionality returns the expected number of dimensions in noiseless data
    num_samples = 100
    num_features = 5
    num_true_dims = 3
    Z = np.random.randn(num_samples,num_true_dims)
    X = Z @ np.random.randn(num_true_dims, num_features)
    X_centered = X - X.mean(axis=0)
    num_dims,eigs = get_dimensionality(X)
    assert num_dims == num_true_dims
    assert np.allclose(np.sort(eigs), np.sort(np.linalg.eigvals(X_centered.T @ X_centered)))

    # test that get_dimensionality returns the expected number of dimensions in noisy data
    X_noisy = X + 1e-2*np.var(X)*np.random.randn(*X.shape)
    num_dims,_ = get_dimensionality(X_noisy)
    assert num_dims == num_true_dims

    # test that var_cutoff works as expected
    num_dims,_ = get_dimensionality(X,var_cutoff=0.5)
    assert num_dims < num_true_dims

    # ensure that mean shifts don't alter the dimensionality
    X_shifted = X + 100*np.var(X)*np.random.randn(1,num_features)
    num_dims,eigs = get_dimensionality(X_shifted)
    assert num_dims == num_true_dims
    assert np.allclose(np.sort(eigs), np.sort(np.linalg.eigvals(X_centered.T @ X_centered)))

def test_max_var_rotate():
    num_samples = 15
    num_features = 5
    num_proj_dims = 3

    proj_mat_manifold = pymanopt.manifolds.Stiefel(num_features,num_proj_dims)
    proj_mat = proj_mat_manifold.random_point() 
    X = np.random.randn(num_samples, num_features)
    new_proj_mat = max_var_rotate(proj_mat, X)

    # Test that max_var_rotate returns a matrix with the expected shape
    assert new_proj_mat.shape == (5, 3)

    # test for orthonormality
    assert np.allclose(new_proj_mat.T @ new_proj_mat, np.eye(num_proj_dims))

    # Test that max_var_rotate returns a matrix with the expected variance
    old_proj_X = X @ proj_mat
    _,sing_vals,_ = np.linalg.svd(old_proj_X-old_proj_X.mean(axis=0),full_matrices=False)
    assert np.allclose(sing_vals**2, num_samples * np.var(X @ new_proj_mat, axis=0))
