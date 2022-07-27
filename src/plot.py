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
            event_time = bins_to_time(trial[event])
            ax.plot(event_time*np.array([1,1]),[0,trial[sig].shape[1]],'--r')

    ax.set_yticks([])
    sns.despine(ax=ax,left=True,bottom=False,trim=True,offset=10)

    return ax