#!/bin/python3

# This script combines all Prez data files into one large multi-session pandas dataframe
# Currently there are three useful Prez datafiles:
# - 20220720
# - 20220721
# - 20220722

#%% setup
import src
import pyaldata
import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA

#%% load data
params = {
    'verbose': True,
    'keep_unsorted': True,
    'bin_size': 0.001,
    'firing_rates_func': lambda td: pyaldata.add_firing_rates(td,method='smooth',std=0.05,backend='convolve'),
    'epoch_fun': src.util.generate_realtime_epoch_fun(
        start_point_name='idx_ctHoldTime',
        end_point_name='idx_endTime',
    ),
}

def load_prez_file(filename):
    td = (
        src.data.load_clean_data(filename,**params)
        .query('task=="RTT" | task=="CST"')
        .astype({'idx_pretaskHoldTime': int})
        .pipe(pyaldata.soft_normalize_signal,signals=['M1_rates','PMd_rates','MC_rates'])
        .pipe(pyaldata.dim_reduce,PCA(n_components=15),'M1_rates','M1_pca')
        .pipe(pyaldata.dim_reduce,PCA(n_components=15),'PMd_rates','PMd_pca')
        .pipe(pyaldata.dim_reduce,PCA(n_components=15),'MC_rates','MC_pca')
    )
    return td

# monkeys = ['Prez']
dates = ['20220720','20220721','20220722']
filenames = [f'../data/trial_data/Prez_{date}_RTTCST_TD.mat' for date in dates]
td_list = pd.concat([load_prez_file(filename) for filename in filenames])
# %%
