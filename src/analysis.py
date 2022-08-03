#!/bin/python3

# library of functions to deal with analysis of data

import numpy as np
import scipy.signal as signal

def normalized_cross_correlation(x, y, **kwargs):
    '''
    Calculate the normalized cross-correlation between two signals, using scipy.signal.correlate.
    This function normalizes the correlation at each lag by the number of points within the valid
    valid window (i.e. it masks out the zero-padded portion to get the normalization)
    
    Args:
        x (array): first signal
        y (array): second signal
        **kwargs: keyword arguments to be passed to scipy.signal.correlate

    Returns:
        array: normalized cross-correlation
    '''
    
    norm_factor = (
        signal.correlate(np.ones(x.shape), np.ones(y.shape), **kwargs)
        *
        np.sqrt(np.mean(x**2) * np.mean(y**2))
    )

    return signal.correlate(x, y, **kwargs) / norm_factor