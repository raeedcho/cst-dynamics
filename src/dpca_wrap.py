import pyaldata
import numpy as np
from dPCA.dPCA import dPCA

import matplotlib.pyplot as plt
import seaborn as sns

def form_neural_tensor(td,signal,cond_cols=None):
    '''
    Form a tensor of neural data from a trial_data structure
    Notes:
        - Trials must be the same length

    Arguments:
        td (DataFrame): trial_data structure in PyalData format
        signal (str): name of signal to form tensor from
        cond_cols (str or list): list of columns to use as conditions to group by
            If None, will form a third order tensor of shape (n_trials,n_neurons,n_timebins)

    Returns:
        np.array: tensor of neural data of shape (n_trials,n_neurons,n_cond_1,n_cond_2,n_cond_3,...,n_timebins)
    '''
    # Argument checking
    assert signal in td.columns, 'Signal must be in trial_data'
    assert pyaldata.trials_are_same_length(td), 'Trials must be the same length'

    if cond_cols is None:
        neural_tensor = np.stack([sig.T for sig in td[signal]],axis=0)
    else:
        td_grouped = td.groupby(cond_cols)
        min_trials = td_grouped.size().min()
        trial_cat_table = td_grouped.agg(
            signal = (signal, lambda sigs: np.stack([sig.T for sig in sigs.sample(n=min_trials)],axis=0))
        )
        # Stack in the remaining axes by iteratively grouping by remaining columns and stacking
        while trial_cat_table.index.nlevels > 1:
            # group by all columns other than the first one
            part_group = trial_cat_table.groupby(level=list(range(1,trial_cat_table.index.nlevels)))
            # stack over the first column and overwrite the old table, keeping time axis at the end
            trial_cat_table = part_group.agg(
                signal = ('signal', lambda sigs: np.stack(sigs,axis=-2))
            )
        else:
            # final stack
            neural_tensor = np.stack(trial_cat_table['signal'],axis=-2)

    return neural_tensor

def plot_dpca(td,latent_dict):
    timevec = (np.arange(td['M1_rates'].values[0].shape[0])-td['idx_goCueTime'].values[0])*td['bin_size'].values[0]
    fig,ax = plt.subplots(2,5,figsize=(15,7),sharex=True,sharey=True)

    for condnum,(key,val) in enumerate(latent_dict.items()):
        for component in range(5):
            for s in range(val.shape[1]):
                ax[condnum,component].plot(timevec,val[component,s,:])

    ax[-1,0].set_xlabel('Time from go cue (s)')
    ax[0,0].set_ylabel('Time-related (Hz)')
    ax[1,0].set_ylabel('Target-related (Hz)')
    for component in range(5):
        ax[0,component].set_title(f'Component #{component+1}')

    sns.despine(fig=fig,trim=True)

    return fig

def plot_dpca_projection(td,array):
    '''
    Plot out dPCA projections from td
    '''
    timevec = (np.arange(td['M1_rates'].values[0].shape[0])-td['idx_goCueTime'].values[0])*td['bin_size'].values[0]
    fig,ax = plt.subplots(2,5,figsize=(15,7),sharex=True,sharey=True)

    # tgt_colors = {0: '#1f77b4', 90: '#ff7f0e', 180: '#2ca02c', -90: '#d62728'}
    task_colors = {'CST': '#1f77b4', 'RTT': '#ff7f0e'}
    for condnum,val in enumerate(['time','target']):
        # plot top five components of each marginalization
        for component in range(5):
            # # group by target direction
            # for tgtDir,td_temp in td.groupby('tgtDir'):
            # plot each trial
            for _,trial in td.iterrows():
                ax[condnum,component].plot(
                    timevec,
                    trial[array+'_dpca_'+val][:,component],
                    color=task_colors[trial['task']],
                    alpha=0.3,
                )

    # pretty up
    ax[-1,0].set_xlabel('Time from go cue (s)')
    ax[0,0].set_ylabel('Time-related (Hz)')
    ax[1,0].set_ylabel('Target-related (Hz)')
    for component in range(5):
        ax[0,component].set_title(f'Component #{component+1}')
    sns.despine(fig=fig,trim=True)

    return fig

def plot_cis_traces(td,array):
    '''
    Plot out top two time-related components of dPCA for all trials
    TODO: make this general by asking for a signal and plotting all trials and average
    '''
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    for _,trial in td.iterrows():
        ax.plot(
            trial[array+'_dpca_time'][:,0],
            trial[array+'_dpca_time'][:,1],
            color=[0.5,0.5,0.5,0.2],
        )

    avg_trace = np.mean(np.stack(td[array+'_dpca_time'],axis=0),axis=0)
    ax.plot(
        avg_trace[:,0],
        avg_trace[:,1],
        color='k',
        linewidth=2,
    )

    ax.set_xlabel('Time component #1 (Hz)')
    ax.set_ylabel('Time component #2 (Hz)')
    sns.despine(fig=fig,trim=True)

    return fig
