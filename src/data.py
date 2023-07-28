from pathlib import Path
import pandas as pd
import pyaldata
import numpy as np
import scipy
import lfads_tf2.utils
from . import lfads_helpers

def load_clean_data(
    file_prefix,
    epoch_fun = lambda trial: slice(0,trial['hand_pos'].shape[0]),
    preload_params = None,
    chop_merge_params = None,
    ):
    """
    Loads and cleans COCST trial data, given a file query
    inputs:
        filepath: full path to file to be loaded

    TODO: set up an initial script to move data into the 'data' folder of the project (maybe with DVC)
    """
    
    assert preload_params is not None, "preload_params must be specified"

    td = preload_data(file_prefix=file_prefix,**preload_params)

    if chop_merge_params is not None:
        td = lfads_helpers.add_lfads_data_to_td(td,file_prefix=file_prefix,**chop_merge_params)

    td = (
        td
        # do sequential trimming to maximally avoid edge effects from kinematic signals
        .pipe(trim_nans, ref_signals=['rel_hand_pos'])
        .pipe(fill_kinematic_signals)
        .pipe(trim_nans,ref_signals=['lfads_rates'])
        .pipe(
            pyaldata.restrict_to_interval,
            epoch_fun = epoch_fun,
            warn_per_trial=True,
        )
        .pipe(add_trial_time,ref_event='idx_goCueTime',column_name='Time from go cue (s)')
        .pipe(add_trial_time,ref_event='idx_pretaskHoldTime',column_name='Time from task cue (s)')
    )

    return td

def preload_data(
    file_prefix,
    verbose=False,
    keep_unsorted=False,
    bin_size=0.01,
    firing_rates_func= lambda td: pyaldata.add_firing_rates(td,method='smooth',std=0.05,backend='convolve'),
):
    datapath = Path('../data/trial_data/')
    filenames = list(datapath.glob(f'{file_prefix}*.mat'))
    if len(filenames) == 0:
        raise FileNotFoundError(f"No files found matching {file_prefix}")
    elif len(filenames) > 1:
        raise ValueError(f"Multiple files found matching {file_prefix}")

    td = (
        pyaldata.mat2dataframe(
            filenames[0],
            shift_idx_fields=True,
            td_name='trial_data'
        )
        .assign(
            date_time=lambda x: pd.to_datetime(x['date_time']),
            session_date=lambda x: pd.DatetimeIndex(x['date_time']).normalize(),
            idx_ctHoldTime= lambda x: x['idx_ctHoldTime'].map(lambda y: y[-1] if type(y) is np.ndarray and y.size>1 else y),
            idx_goCueTime= lambda x: x['idx_goCueTime'].map(lambda y: y[0] if type(y) is np.ndarray and y.size>1 else y),
        )
        .pipe(remove_aborts, verbose=verbose)
        .astype({
            'idx_ctHoldTime': int,
            'idx_pretaskHoldTime': int,
            'idx_goCueTime': int,
        })
        .pipe(remove_artifact_trials, verbose=verbose)
        .pipe(
            filter_unit_guides,
            filter_func=lambda guide: guide[:,1] >= (0 if keep_unsorted else 1)
        )
        .pipe(remove_correlated_units, verbose=verbose)
        .pipe(
            remove_all_low_firing_neurons,
            threshold=0.1,
            divide_by_bin_size=True,
            verbose=verbose
        )
        .pipe(rebin_data,new_bin_size=bin_size)
        .pipe(firing_rates_func)
    )
    return td

@pyaldata.copy_td
def remove_aborts(td,verbose=False):
    """
    Remove trials that were aborted
    """
    abort_idx = pd.isna(td["idx_goCueTime"])
    td = td[~abort_idx]
    if verbose:
        print(f"Removed {sum(abort_idx)} trials that monkey aborted")
    
    return td

def remove_all_low_firing_neurons(trial_data, signals=None, **kwargs):
    '''
    Removes all low firing neurons from the trial data.
    Wrapper on pyaldata.remove_low_firing_neurons.

    Arguments:
        trial_data: trial data to be filtered
        signals: list of signals to be filtered (if None, any signal ending in '_spikes' will be filtered)
        **kwargs: keyword arguments to be passed to pyaldata.remove_low_firing_neurons
            (except the signal name, which is hard-coded to any column with 'spikes' in the name)
    '''
    if signals is None:
        signals = [s for s in trial_data.columns if s.endswith('_spikes')]

    for sig in signals:
        trial_data = pyaldata.remove_low_firing_neurons(trial_data, sig, **kwargs)

    return trial_data

@pyaldata.copy_td
def trim_nans(trial_data, ref_signals=["rel_hand_pos"]):
    """
    Trim nans off of end of trials when hand position wasn't recorded
    """

    if len(set(ref_signals).intersection(trial_data.columns)) == 0:
        return trial_data

    def epoch_fun(trial):
        signals = np.column_stack([trial[sig] for sig in ref_signals if sig in trial.index])
        nan_times = np.any(np.isnan(signals), axis=1)
        first_viable_time = np.nonzero(~nan_times)[0][0]
        last_viable_time = np.nonzero(~nan_times)[0][-1]
        return slice(first_viable_time, last_viable_time + 1)

    td_trimmed = pyaldata.restrict_to_interval(trial_data,epoch_fun=epoch_fun,reset_index=False)
    for trial_id in td_trimmed.index:
        td_trimmed.loc[trial_id, "idx_endTime"] = np.column_stack(
            [td_trimmed.loc[trial_id,sig] for sig in ref_signals]
        ).shape[0]

    return td_trimmed


@pyaldata.copy_td
def fill_kinematic_signals(td, cutoff=30):
    """
    Fill out kinematic signals by filtering hand position and differentiating
    Inputs:
        - td: PyalData file
        - cutoff: cutoff frequency (Hz) for low-pass filter on hand position (default: 30)
    """
    samprate = 1 / td.at[td.index[0], "bin_size"]
    filt_b, filt_a = scipy.signal.butter(4, cutoff / (samprate / 2))
    td["hand_pos"] = [
        scipy.signal.filtfilt(filt_b, filt_a, signal, axis=0)
        for signal in td["hand_pos"]
    ]
    td["hand_vel"] = [
        np.gradient(trial["hand_pos"], trial["bin_size"], axis=0)
        for _, trial in td.iterrows()
    ]
    td["hand_acc"] = [
        np.gradient(trial["hand_vel"], trial["bin_size"], axis=0)
        for _, trial in td.iterrows()
    ]
    td["cursor_vel"] = [
        np.gradient(trial["cursor_pos"], trial["bin_size"], axis=0)
        for _, trial in td.iterrows()
    ]
    td["hand_speed"] = [np.linalg.norm(vel, axis=1) for vel in td["hand_vel"]]

    return td

@pyaldata.copy_td
def filter_unit_guides(td,filter_func):
    """
    Removes unsorted units from TD, given array name

    Inputs:
        - td: PyalData DataFrame
        - filter_func: function to filter units with (callable that takes unit guide as input and returns boolean array)

    Returns:
        - td: PyalData DataFrame with filtered units
    """
    

    array_names = [name.replace('_spikes', '') for name in td.columns if name.endswith('_spikes')]
    for array in array_names:
        unit_guide = td.loc[td.index[0],f'{array}_unit_guide']

        if unit_guide.shape[0] == 0:
            continue

        # remove unsorted neurons (unit number <=1)
        good_units = filter_func(unit_guide)

        # mask out bad neural data
        td =  mask_neural_data(td, array, good_units)

    return td

@pyaldata.copy_td
def remove_correlated_units(td, arrays=['M1','PMd','MC'], verbose=False):
    '''
    Removes correlated units from TD, given array name

    TODO: change this to remove neuron pairs that have 1ms correlation > 0.2
    '''

    # calculate correlation matrix
    for array in arrays:
        unit_guide = td.loc[td.index[0],f'{array}_unit_guide']
        if unit_guide.shape[0] == 0:
            continue

        # calculate correlation matrix
        corr_mat = np.corrcoef(np.row_stack(td[f'{array}_spikes']).T)

        # find correlated units
        corr_units = np.any(np.tril(corr_mat,k=-1) > 0.2, axis=1)

        # remove correlated units (TODO: possibly only remove one neuron at a time)
        td =  mask_neural_data(td, array, ~corr_units)

        # verbose
        if verbose:
            print(f'{array}: {np.sum(corr_units)} correlated units removed')

    # # for now, these particular files only...
    # if td.loc[td.index[0], "session_date"] == pd.to_datetime("2018/06/26"):
    #     unit_guide = td.loc[td.index[0],'M1_unit_guide']
    #     corr_units = np.array([[8, 2], [64, 2]])
    #     bad_units = (
    #         np.in1d(unit_guide[:, 0], corr_units[:, 0])
    #         & np.in1d(unit_guide[:, 1], corr_units[:, 1])
    #     )
    #     td = mask_neural_data(td, 'M1', ~bad_units)
    # elif td.loc[td.index[0], "session_date"] == pd.to_datetime("2022/07/20"):
    #     corr_units = np.array([[71, 1], [73, 1]])
    #     for array in ['M1','PMd','MC']:
    #         unit_guide = td.loc[td.index[0],f'{array}_unit_guide']
    #         bad_units = (
    #             np.in1d(unit_guide[:, 0], corr_units[:, 0])
    #             & np.in1d(unit_guide[:, 1], corr_units[:, 1])
    #         )
    #         td = mask_neural_data(td,array, ~bad_units)

    return td


@pyaldata.copy_td
def mask_neural_data(td, array, mask):
    """
    Returns a PyalData structure with neural units kept according to mask
    (Basically just masks out both the actual neural data and the unit guide simultaneously)

    Inputs:
        - td: PyalData structure
        - array: name of array to mask
        - mask: boolean array of units to keep

    Outputs:
        - td: PyalData structure with neural units masked out
    """
    td[f'{array}_spikes'] = [spikes[:, mask] for spikes in td[f'{array}_spikes']]
    td[f'{array}_unit_guide'] = [guide[mask, :] for guide in td[f'{array}_unit_guide']]

    return td


@pyaldata.copy_td
def relationize_td(td):
    """
    Split out trial info and time-varying signals into their own tables

    Returns (trial_info,signals)
        trial_info - DataFrame with info about individual trials
        signals - 'trialtime' indexed DataFrame with signal values
    """
    # start out by making sure that trial_id is index
    td = (
        td
        .pipe(add_trial_time)
    )

    # separate out non-time-varying fields into a separate trial table
    timevar_cols = pyaldata.get_time_varying_fields(td)
    trial_info = td.drop(columns=timevar_cols).set_index('trial_id')
    signals = (
        td
        .filter(items=['trial_id']+timevar_cols)
        .explode(timevar_cols)
        .assign(trialtime=lambda x: pd.to_timedelta(x['trialtime'], unit='seconds'))
        .set_index(['trial_id','trialtime'])
    )

    return trial_info, signals

@pyaldata.copy_td
def explode_td(td):
    '''
    Explodes a pyaldata-style dataframe into a normal one, where each row
    is a time point in a trial.

    Arguments:
        - td (pd.DataFrame): dataframe in form of PyalData

    Returns:
        - (pd.DataFrame): exploded dataframe with hierarchical index:
            axis 0: trial id, time bin in trial
            axis 1: signal name
    '''
    timevar_cols = pyaldata.get_time_varying_fields(td)
    exp_td = (
        td
        .pipe(add_trial_time)
        .explode(timevar_cols)
        .assign(trialtime=lambda x: pd.to_timedelta(x['trialtime'], unit='seconds'))
        .set_index(['trial_id','trialtime'])
    )

    return exp_td

def crystalize_dataframe(td,sig_guide=None):
    '''
    Transforms a pyaldata-style dataframe into a normal one, where each row
    is a time point in a trial. This is useful for some manipulations,
    especially those that involve melting the dataframe.

    Arguments:
        - td (pd.DataFrame): dataframe in form of PyalData
        - sig_guide (dict): dictionary of signals with keys corresponding to the signal names
            in the pyaldata dataframe and values corresponding to the names of each column
            of the individual signal in the dataframe. If None, all signals are included,
            with columns labeled 0-N, where N is the number of columns in the signal.

    Returns:
        - (pd.DataFrame): crystallized dataframe with hierarchical index on both axes:
            axis 0: trial id, time bin in trial
            axis 1: signal name, signal dimension
    '''
    # TODO: check that signals are in the dataframe and are valid signals

    if sig_guide is None:
        sig_guide = {sig: None for sig in pyaldata.get_time_varying_fields(td)}

    if type(sig_guide) is list:
        sig_guide = {sig: None for sig in sig_guide}

    assert type(sig_guide) is dict, "sig_guide must be a dictionary"

    df = pd.concat(
        [
            pd.concat([pd.DataFrame(trial[sig],columns=guide) for sig,guide in sig_guide.items()], axis=1, keys=sig_guide.keys()) 
            for _,trial in td.iterrows()
        ],
        axis=0,
        keys=td['trial_id'],
    )
    df.index.rename('Time bin',level=1,inplace=True)
    return df

def extract_metaframe(td,metacols=None):
    '''
    Extracts a metaframe from a trial dataframe.

    Arguments:
        - td (pd.DataFrame): dataframe in form of PyalData
        - metacols (list of str): columns to include in the metaframe
            Default behavior (if None) is to get all columns that are not time-varying signals

    Returns:
        - (pd.DataFrame): metaframe with hierarchical index on both axes:
            axis 0: trial id
            axis 1: column name
    '''
    if metacols is None:
        metacols = set(td.columns) - set(pyaldata.get_time_varying_fields(td))

    meta_df = td.filter(items=metacols).set_index('trial_id')
    meta_df.columns = pd.MultiIndex.from_product([meta_df.columns,['meta']])
    return meta_df

def extract_unit_guides(td,array='MC'):
    '''
    Extracts a unit guide dataframe from a trial data dataframe.

    Note: if this is Prez data, all we need is MC unit guide, since other arrays
    are subsets of MC (M1 is channels 33-96 and PMd is channels 1-32;97-128).

    Arguments:
        - td (pd.DataFrame): dataframe in form of PyalData
        - array (str): name of array to extract unit guide from

    Returns:
        - (pd.DataFrame): unit guide dataframe with index 0-N, where N is the number
            of units in the array and columns {'channel','unit','array'}
    '''
    
    if td['monkey'].iloc[0] == 'Prez':
        def which_array(channel):
            if 32<channel<=96:
                return 'M1'
            else:
                return 'PMd'
    else:
        Warning('Array splitting only defined for Prez data currently.')
        def which_array(channel):
            return array

    unit_guide = (
        pd.DataFrame(
            td[f'{array}_unit_guide'].values[0],
            columns=['channel','unit'],
        )
        .assign(array=lambda x: x['channel'].apply(which_array))
    )
    return unit_guide

def extract_trial_events(td,events=None):
    '''
    Extracts a trial events dataframe from a trial data dataframe.

    Arguments:
        - td (pd.DataFrame): dataframe in form of PyalData
        - events (list of str): events to include in the trial events dataframe
            Default behavior (if None) is to get all events in the dataframe

    Returns:
        - (pd.DataFrame): trial events dataframe with hierarchical index on both axes:
            axis 0: trial id
            axis 1: event name
    '''
    if events is None:
        events = set(td.columns) - set(pyaldata.get_time_varying_fields(td))

    events_df = td.filter(items=events).set_index('trial_id')
    events_df.columns = pd.MultiIndex.from_product([events_df.columns,['event']])
    return events_df

@pyaldata.copy_td
def add_trial_time(trial_data, ref_event=None, column_name="trialtime"):
    """
    Add a trialtime column to trial_data, based on the bin_size and shape of hand_pos

    Arguments:
        - trial_data: DataFrame in form of PyalData
        - ref_event: string indicating which event to use as reference for trial time
            (e.g. 'idx_goCueTime') (default: None)

    Returns:
        - trial_data: DataFrame with trialtime column added
    """
    if ref_event is None:
        trial_data[column_name] = [
            trial["bin_size"] * np.arange(trial["hand_pos"].shape[0])
            for (_, trial) in trial_data.iterrows()
        ]
    else:
        trial_data[column_name] = [
            trial["bin_size"]
            * (np.arange(trial["hand_pos"].shape[0]) - trial[ref_event])
            for (_, trial) in trial_data.iterrows()
        ]

    return trial_data

@pyaldata.copy_td
def remove_artifact_trials(trial_data, rate_thresh=350,verbose=False):
    '''
    Remove trials with neural artifacts (mostly in Prez when he moves his head).
    These artifacts look like very high firing rates for a hundred milliseconds or so.

    Arguments:
        - trial_data: DataFrame in form of PyalData
        - rate_thresh: threshold for firing rate (Hz)
        - verbose: print out information about removed trials

    Returns:
        - trial_data: DataFrame with bad trials removed
    '''

    # bin trial data at 100 ms (if it's not already binned so much)
    if all(trial_data['bin_size']<=0.05): # if bin_size is already greater than 0.05, don't bin
        td_temp = pyaldata.combine_time_bins(trial_data, n_bins=int(0.1/trial_data['bin_size'].values[0]))
    else:
        td_temp = trial_data.copy()

    td_temp = pyaldata.add_firing_rates(td_temp,method='bin')
    array_names = [name.replace('_rates', '') for name in td_temp.columns if name.endswith('_rates')]

    # find trials with high firing rates
    # for id, trial in td_temp.iterrows():
    #     for array_name in array_names:
    #         if any(trial[f'{array_name}_rates']>rate_thresh):
    #             bad_trials.append(id)
    #             break

    bad_trials = {
        id
        for id,trial in td_temp.iterrows()
        for array_name in array_names
        if (trial[f'{array_name}_rates']>rate_thresh).any()
    }

    if verbose:
        print(f'{len(bad_trials)} trials with high firing rates removed. Dropping trials with IDs:')
        print(td_temp.loc[list(bad_trials),"trial_id"].values)

    return trial_data.drop(index=bad_trials)
    
def rebin_data(trial_data, new_bin_size):
    '''
    Re-bins the trial data at a new bin size.
    
    Note: this is a wrapper on pyaldata.combine_time_bins.
    '''
    assert new_bin_size >= trial_data['bin_size'].values[0], "New bin size must be larger than old bin size."
    assert np.isclose(new_bin_size % trial_data['bin_size'].values[0],0), "New bin size must be a multiple of old bin size."

    if new_bin_size==trial_data['bin_size'].values[0]:
        return trial_data.copy()
    else:
        return pyaldata.combine_time_bins(trial_data, n_bins=int(new_bin_size/trial_data['bin_size'].values[0]))

@pyaldata.copy_td
def remove_baseline_rates(trial_data, signals=['MC_rates']):
    '''
    Removes the baseline firing rate from the signal, where "baseline"
    is defined as the average firing rate from -300ms to -50ms from
    when the task cue is presented.
    '''

    if type(signals)==str:
        signals = [signals]

    for signal in signals:
        # get baseline firing rates
        baseline_rates = (
            trial_data.groupby('result').get_group('R')
            .filter(items=[
                'trial_id',
                'Time from task cue (s)',
                signal,
            ])
            .explode([
                'Time from task cue (s)',
                signal,
            ])
            .astype({
                'Time from task cue (s)': 'float',
            })
            .query('`Time from task cue (s)`>=-0.3 and `Time from task cue (s)`<=-0.05')
            [signal]
            .mean()
        )

        # subtract baseline firing rates
        trial_data[signal] = trial_data[signal].apply(lambda x: x-baseline_rates)

    return trial_data