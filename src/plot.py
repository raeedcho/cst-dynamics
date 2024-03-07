from configparser import Interpolation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from . import subspace_tools

def make_trial_raster(trial, ax=None, sig='M1_spikes', events=None, ref_event_idx=0):
    '''
    Make a raster plot for a given trial

    Arguments:
        trial - pandas Series with trial_data structure (a row of trial_data)
        ax - matplotlib axis to plot into (default: None)
        events - list of event times to plot (default: None)
        ref_event_idx - event index to reference the trial time to (default: 0)

    Returns:
        ax - matplotlib axis with traces plotted
    '''
    if ax is None:
        ax = plt.gca()
        
    ax.clear()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')

    def bins_to_time(bins):
        return (bins-ref_event_idx)*trial['bin_size']

    if trial['bin_size'] > 0.01:
        # use an image to plot the spikes
        time_extent = (bins_to_time(0), bins_to_time(trial[sig].shape[0]))
        neuron_extent = (-0.5, trial[sig].shape[1]-0.5)
        ax.imshow(
            trial[sig].T,
            aspect='auto',
            cmap='binary',
            origin='lower',
            extent=time_extent+neuron_extent,
            interpolation='none',
        )
    else:
        # use sparse matrix plot
        spike_bins,spike_neurons = np.nonzero(trial[sig])
        ax.plot(
            bins_to_time(spike_bins),
            spike_neurons,
            '|k',
            markersize=1,
        )

    if events is not None:
        for event in events:
            event_times = bins_to_time(trial[event])
            event_times = event_times[~np.isnan(event_times)]
            if event_times.size == 1:
                ax.plot(
                    event_times * np.array([1,1]),
                    [0,trial[sig].shape[1]],
                    '--r'
                )
                ax.text(
                    event_times,
                    trial[sig].shape[1]+2,
                    event.replace('idx_',''),
                    color='b',
                    fontsize=14,
                    fontweight='black',
                    rotation='vertical',
                    verticalalignment='top',
                )
            elif event_times.size > 1:
                for ev_t in event_times:
                    ax.plot(
                        ev_t * np.array([1,1]),
                        [0,trial[sig].shape[1]],
                        '--r'
                    )
                    ax.text(
                        ev_t,
                        trial[sig].shape[1]+2,
                        event.replace('idx_',''),
                        color='b',
                        fontsize=14,
                        fontweight='black',
                        rotation='vertical',
                        verticalalignment='top',
                    )

    ax.set_yticks([])
    sns.despine(ax=ax,left=True,bottom=False,trim=True,offset=10)

    return ax

def plot_hand_trace(trial,ax=None,timesig='trialtime',trace_component=0):
    if ax is None:
        ax = plt.gca()

    # zero line
    ax.plot([trial[timesig][0],trial[timesig][-1]],[0,0],'-k')

    # targets
    targ_size = 10
    if not np.isnan(trial['idx_ctHoldTime']) and not np.isnan(trial['idx_pretaskHoldTime']):
        ax.add_patch(Rectangle(
            (trial[timesig][trial['idx_ctHoldTime']],-targ_size/2),
            trial[timesig][trial['idx_pretaskHoldTime']]-trial[timesig][trial['idx_ctHoldTime']],
            targ_size,
            color='0.5',
        ))

    if trial['task']=='RTT':
        if not np.isnan(trial['idx_pretaskHoldTime']) and not np.isnan(trial['idx_goCueTime']):
            ax.add_patch(Rectangle(
                (trial[timesig][trial['idx_pretaskHoldTime']],-targ_size/2),
                trial[timesig][trial['idx_goCueTime']]-trial[timesig][trial['idx_pretaskHoldTime']],
                targ_size,
                color='C1',
            ))
        for idx_targ_start,idx_targ_end,targ_loc in zip(
            trial['idx_rtgoCueTimes'].astype(int),
            trial['idx_rtHoldTimes'].astype(int),
            trial['rt_locations'][:,trace_component]-trial['ct_location'][trace_component],
        ):
            if not np.isnan(idx_targ_start) and not np.isnan(idx_targ_end):
                ax.add_patch(Rectangle(
                    (trial[timesig][idx_targ_start],targ_loc-targ_size/2),
                    trial[timesig][idx_targ_end]-trial[timesig][idx_targ_start],
                    targ_size,
                    color='C1',
                ))
    elif trial['task']=='CST':
        if not np.isnan(trial['idx_pretaskHoldTime']) and not np.isnan(trial['idx_goCueTime']):
            ax.add_patch(Rectangle(
                (trial[timesig][trial['idx_pretaskHoldTime']],-targ_size/2),
                trial[timesig][trial['idx_goCueTime']]-trial[timesig][trial['idx_pretaskHoldTime']],
                targ_size,
                color='C0',
            ))

    # cursor
    ax.plot(
        trial[timesig],
        trial['rel_cursor_pos'][:,trace_component],
        c='b',
        alpha=0.5,
    )
    
    # hand
    ax.plot(
        trial[timesig],
        trial['rel_hand_pos'][:,trace_component],
        c='k',
    )
    ax.set_ylim(-60,60)
    ax.set_ylabel('Hand position (cm)')
    ax.set_xlabel(timesig)
    sns.despine(ax=ax,trim=True)

def plot_hand_acc(trial,ax=None,timesig='trialtime',trace_component=0):
    if ax is None:
        ax = plt.gca()

    ax.plot([trial[timesig][0],trial[timesig][-1]],[0,0],'-k')
    ax.plot(
        trial[timesig],
        trial['hand_acc'][:,trace_component],
        color='k',
    )

    ax.set_ylabel('Hand acceleration (cm/s^2)')
    ax.set_xlabel(timesig)
    sns.despine(ax=ax,trim=True)

def plot_hand_velocity(trial,ax=None,timesig='trialtime',trace_component=0):
    if ax is None:
        ax = plt.gca()

    ax.plot([trial[timesig][0],trial[timesig][-1]],[0,0],'-k')
    ax.plot(
        trial[timesig],
        trial['hand_vel'][:,trace_component],
        color='k',
    )

    ax.set_ylabel('Hand velocity (cm/s)')
    ax.set_xlabel(timesig)
    sns.despine(ax=ax,trim=True)

def plot_split_subspace_variance(td,signal='lfads_rates_joint_pca'):
    compared_var = (
        td
        .groupby('task')
        [[f'{signal}',f'{signal}_split']]
        .agg([
            lambda s,col=col: subspace_tools.calculate_fraction_variance(np.row_stack(s),col)
            for col in range(40)
        ])
        .rename({f'{signal}': 'unsplit',f'{signal}_split': 'split'},axis=1,level=0)
        .rename(lambda label: label.strip('<lambda_>'),axis=1,level=1)
        .unstack()
        .reset_index()
        .rename({
            'level_0': 'neural space',
            'level_1':'component',
            0: 'fraction variance'
        },axis=1)
    )

    sns.catplot(
        data=compared_var,
        y='component',
        x='fraction variance',
        hue='task',
        kind='bar',
        col='neural space',
        sharex=True,
        sharey=True,
        aspect=0.5,
        height=6,
    )

import plotly.express as px
def plot_single_trial_split_var(td,signal='lfads_rates_joint_pca'):
    subspace_var = (
        td
        .assign(**{
            'CST space variance': lambda df: df.apply(
                lambda row: np.var(row[f'{signal}_cst_unique'],axis=0).sum()/np.var(row[f'{signal}'],axis=0).sum(),
                axis=1,
            ),
            'RTT space variance': lambda df: df.apply(
                lambda row: np.var(row[f'{signal}_rtt_unique'],axis=0).sum()/np.var(row[f'{signal}'],axis=0).sum(),
                axis=1,
            ),
        })
    )
    
    fig = px.scatter(
        subspace_var,
        x='CST space variance',
        y='RTT space variance',
        color='task',
        hover_data=['trial_id'],
        marginal_x='violin',
        marginal_y='violin',
        template='plotly_white',
        width=600,
        height=600,
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    fig.show()
    print(subspace_var.set_index('trial_id').loc[11,['task','CST space variance','RTT space variance']])

    # return sns.relplot(
    #     data=subspace_var,
    #     x='CST space variance',
    #     y='RTT space variance',
    #     hue='task',
    #     hue_order=['CST','RTT'],
    # )