from configparser import Interpolation
import numpy as np
import matplotlib.pyplot as plt
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

def plot_hand_trace(trial,ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot([0,trial['trialtime'][-1]],[0,0],'-k')
    ax.plot(
        trial['trialtime'],
        trial['rel_cursor_pos'][:,0],
        c='b',
        alpha=0.25,
    )
    ax.plot(
        trial['trialtime'],
        trial['rel_hand_pos'][:,0],
        c='k',
    )
    ax.set_ylim(-60,60)
    ax.set_ylabel('Hand position')
    ax.set_xlabel('Time after go cue (s)')
    sns.despine(ax=ax,trim=True)