from configparser import Interpolation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

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

def plot_hand_trace(trial,ax=None,timesig='trialtime'):
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
            trial['rt_locations'][:,0]-trial['ct_location'][0],
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
        trial['rel_cursor_pos'][:,0],
        c='b',
        alpha=0.5,
    )
    
    # hand
    ax.plot(
        trial[timesig],
        trial['rel_hand_pos'][:,0],
        c='k',
    )
    ax.set_ylim(-60,60)
    ax.set_ylabel('Hand position (cm)')
    ax.set_xlabel(timesig)
    sns.despine(ax=ax,trim=True)

def plot_hand_velocity(trial,ax=None,timesig='trialtime'):
    if ax is None:
        ax = plt.gca()

    ax.plot([trial[timesig][0],trial[timesig][-1]],[0,0],'-k')
    ax.plot(
        trial[timesig],
        trial['hand_vel'][:,0],
        color='k',
    )

    ax.set_ylabel('Hand position (cm)')
    ax.set_xlabel(timesig)
    sns.despine(ax=ax,trim=True)
