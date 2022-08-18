#!/bin/python3

# library of functions to deal with analysis of data

import numpy as np
import scipy.signal as signal

def normalized_cross_correlation(x, y, **kwargs):
    '''
    Calculate the normalized cross-correlation between two signals, using scipy.signal.correlate.
    This function normalizes the correlation at each lag by the number of points within the valid
    valid window (i.e. it masks out the zero-padded portion to get the normalization). The normalization
    factor is the product of the root mean squares of the input signals (which is the same as the 
    auto-correlation at lag 0). Also, we want to mean subtract the two signals before calculating
    the correlation to make things match up with a Pearson's r type of intution.
    
    Args:
        x (array): first signal
        y (array): second signal
        **kwargs: keyword arguments to be passed to scipy.signal.correlate

    Returns:
        array: normalized cross-correlation
    '''
    x_hat = x-np.mean(x)
    y_hat = y-np.mean(y)
    
    norm_factor = (
        signal.correlate(np.ones(x.shape), np.ones(y.shape), **kwargs)
        *
        np.sqrt(np.mean(x_hat**2) * np.mean(y_hat**2))
    )

    return signal.correlate(x_hat, y_hat, **kwargs) / norm_factor