#!/bin/python3

'''
This script loads a trial data file and checks for coincident spike artifacts.

The way it checks for these artifacts is by calculating the fraction of channels in the array
that are active simultaneously in 10 ms bins. If the fraction is greater than 0.5, then the
that time bin is considered to be an artifact.
'''

#%%
import src
import numpy as np
import pyaldata
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')

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

filename = '../data/trial_data/Prez_20220721_RTTCST_TD.mat'
td = src.data.load_clean_data(filename,**params)

# %%
# concatenate all spikes into one array and calculate the fraction of channels that are active
spikes = np.row_stack(td['MC_spikes'])
frac_coincident = np.mean(spikes, axis=1)

# %%
# plot the fraction of channels that are active
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(frac_coincident.shape[0])*params['bin_size'],frac_coincident)
ax.set_ylim([0,1])
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'Fraction of channels active\nin {params["bin_size"]*1000} ms bin')
sns.despine(ax=ax,trim=True)


# %% plot the 1ms correlation matrix of spikes
corr = np.corrcoef(spikes.T)
fig,ax = plt.subplots(1,1)
sns.heatmap(np.tril(corr,k=-1),ax=ax)
ax.set_xlabel('Unit')
ax.set_ylabel('Unit')
sns.despine(ax=ax,trim=True)


# %%
