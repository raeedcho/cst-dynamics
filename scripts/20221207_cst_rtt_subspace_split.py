#!/bin/python3

'''
This script is meant to see if CST and RTT have unique, non-shared dimensions of neural activity.

It uses Brian's Matlab script to split subspaces, so partway through, this script exports 
'''

#%% Setup
import src
import pyaldata
import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit

from ipywidgets import interact
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle,Circle
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
sns.set_context('talk')

params = {
    'verbose': True,
    'keep_unsorted': True,
    'bin_size': 0.010,
    'firing_rates_func': lambda td: pyaldata.add_firing_rates(td,method='smooth',std=0.05,backend='convolve'),
    'epoch_fun': src.util.generate_realtime_epoch_fun(
        start_point_name='idx_ctHoldTime',
        end_point_name='idx_endTime',
    ),
}

filename = '../data/trial_data/Prez_20220720_RTTCST_TD.mat'
td = (
    src.data.load_clean_data(
        filename,
        **params
    )
    .query('task=="RTT" | task=="CST"')
    .astype({'idx_pretaskHoldTime': int})
    .pipe(pyaldata.soft_normalize_signal,signals=['M1_rates','PMd_rates','MC_rates'])
    .pipe(pyaldata.dim_reduce,PCA(n_components=15),'M1_rates','M1_pca')
    .pipe(pyaldata.dim_reduce,PCA(n_components=15),'PMd_rates','PMd_pca')
    .pipe(pyaldata.dim_reduce,PCA(n_components=15),'MC_rates','MC_pca')
    .assign(
        beh_sig=lambda x: x.apply(
            lambda y: np.column_stack([
                y['rel_hand_pos'][:,[0,1,2]],
                y['hand_vel'][:,:],
                y['hand_acc'][:,:],
            ]),axis=1
        )
    )
)