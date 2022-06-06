import pandas as pd
import pyaldata
import numpy as np
import scipy
import h5py
import os
from sklearn.model_selection import train_test_split

from lfads_tf2.utils import chop_data, merge_chops


def load_clean_data(filepath, verbose=False):
    """
    Loads and cleans COCST trial data, given a file query
    inputs:
        filepath: full path to file to be loaded

    TODO: set up an initial script to move data into the 'data' folder of the project (maybe with DVC)
    """
    td = pyaldata.mat2dataframe(filepath, True, "trial_data")

    # condition dates and times
    td["date_time"] = pd.to_datetime(td["date_time"])
    td["session_date"] = pd.DatetimeIndex(td["date_time"]).normalize()

    # remove aborts
    abort_idx = np.isnan(td["idx_goCueTime"])
    td = td[~abort_idx]
    if verbose:
        print(f"Removed {sum(abort_idx)} trials that monkey aborted")

    # td = trim_nans(td)
    # td = fill_kinematic_signals(td)

    # neural data considerations
    unit_guide = td.loc[td.index[0], "M1_unit_guide"]
    if unit_guide.shape[0] > 0:
        # remove unsorted neurons (unit number <=1)
        bad_units = unit_guide[:, 1] <= 1

        # for this particular file only, remove correlated neurons...
        if td.loc[td.index[0], "session_date"] == pd.to_datetime("2018/06/26"):
            corr_units = np.array([[8, 2], [64, 2]])
            bad_units = bad_units | (
                np.in1d(unit_guide[:, 0], corr_units[:, 0])
                & np.in1d(unit_guide[:, 1], corr_units[:, 1])
            )

        # mask out bad neural data
        td["M1_spikes"] = [spikes[:, ~bad_units] for spikes in td["M1_spikes"]]
        td["M1_unit_guide"] = [guide[~bad_units, :] for guide in td["M1_unit_guide"]]

        td = pyaldata.remove_low_firing_neurons(
            td, "M1_spikes", 0.1, divide_by_bin_size=True, verbose=verbose
        )

    return td


def trim_nans(trial_data, ref_signals=["rel_hand_pos"]):
    """
    Trim nans off of end of trials when hand position wasn't recorded
    """

    def epoch_fun(trial):
        signals = np.column_stack([trial[sig] for sig in ref_signals])
        nan_times = np.any(np.isnan(signals), axis=1)
        first_viable_time = np.nonzero(~nan_times)[0][0]
        last_viable_time = np.nonzero(~nan_times)[0][-1]
        return slice(first_viable_time, last_viable_time + 1)

    td_trimmed = pyaldata.restrict_to_interval(trial_data, epoch_fun=epoch_fun)
    for trial_id in td_trimmed.index:
        td_trimmed.loc[trial_id, "idx_endTime"] = td_trimmed.loc[
            trial_id, "M1_spikes"
        ].shape[0]

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
def mask_neural_data(td, mask):
    """
    Returns a PyalData structure with neural units kept according to mask
    (Basically just masks out both the actual neural data and the unit guide simultaneously)
    """
    pass


@pyaldata.copy_td
def relationize_td(trial_data):
    """
    Split out trial info and time-varying signals into their own tables

    Returns (trial_info,signals)
        trial_info - DataFrame with info about individual trials
        signals - 'trialtime' indexed DataFrame with signal values
    """
    # start out by making sure that trial_id is index
    td = trial_data.set_index("trial_id")

    # separate out non-time-varying fields into a separate trial table
    timevar_cols = td.columns.intersection(pyaldata.get_time_varying_fields(td))
    trial_info = td.drop(columns=timevar_cols)
    timevar_data = td[timevar_cols].copy()

    # melt out time information in time-varying column dataframe
    signals = []
    for (idx, trial) in timevar_data.iterrows():
        # split out rows of numpy array
        signal_dict = {
            key: list(val_array.copy()) for (key, val_array) in trial.iteritems()
        }
        signal_dict["trial_id"] = idx
        temp = pd.DataFrame(signal_dict)

        # add a timedelta column to DataFrame
        # temp['trialtime'] = pd.to_timedelta(trial_info.loc[idx,'bin_size']*np.arange(trial[timevar_cols[0]].shape[0]))
        temp["trialtime"] = pd.to_timedelta(
            trial_info.loc[idx, "bin_size"]
            * np.arange(trial[timevar_cols[0]].shape[0]),
            unit="seconds",
        )

        signals.append(temp)

    signals = pd.concat(signals)
    signals.set_index(["trial_id", "trialtime"], inplace=True)

    # set up a multi-index for trials
    # td.set_index(['monkey','session_date','trial_id'],inplace=True)

    return trial_info, signals


@pyaldata.copy_td
def add_trial_time(trial_data, ref_event=None):
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
        trial_data["trialtime"] = [
            trial["bin_size"] * np.arange(trial["hand_pos"].shape[0])
            for (_, trial) in trial_data.iterrows()
        ]
    else:
        trial_data["trialtime"] = [
            trial["bin_size"]
            * (np.arange(trial["hand_pos"].shape[0]) - trial[ref_event])
            for (_, trial) in trial_data.iterrows()
        ]

    return trial_data


def prep_neural_tensors(
    trial_data, signal="M1_spikes", bin_size=0.002, window_len=350, overlap=0
):
    """
    Creates a tensor of neural data for each trial by chopping up the neural data
    into overlapping windows.

    Arguments:
        - trial_data (pd.DataFrame): DataFrame in form of PyalData
        - signal (string): which signal to use (default: 'M1_spikes')
        - bin_size (float): size of each bin in seconds(default: 0.002)
        - window_len (int): number of bins in each window (default: 350)
        - overlap (int): number of overlap bins between windows (default: 0)

    Returns:
        - (list of np.ndarray): list of tensors of neural data, one for each trial.
            Shape of each tensor will be (num_windows, window_len, num_units)
        - (np.ndarray): array of trial ids for each tensor
    """
    # bin data if necessary
    if np.isclose(bin_size, trial_data["bin_size"].values[0]):
        td = trial_data
    elif bin_size > 1.9 * trial_data["bin_size"].values[0]:
        td = pyaldata.combine_time_bins(
            trial_data, int(bin_size / trial_data["bin_size"].values[0])
        )
    else:
        raise ValueError("bin_size must be greater than trial_data.bin_size")

    # compose tensors
    tensor_list = [chop_data(sig, overlap, window_len) for sig in td[signal]]
    trial_ids = np.array(td["trial_id"])

    return tensor_list, trial_ids


def write_tensor_to_hdf(tensor_list, trial_ids, filename):
    """
    Writes a neural tensor to an hdf5 file, splitting into train and validation sets.

    Arguments:
        - tensor_list (list of np.ndarray): list of tensors (one for each trial) to concatenate and write to file
            (shape of each tensor: (num_windows, window_len, num_units))
        - filename (string): name of file to write to
    """

    # clear file if it exists
    try:
        os.remove(filename)
        print("Overwriting file: {}".format(filename))
    except OSError:
        pass

    trial_id_map = np.concatenate(
        [id * np.ones(tensor.shape[0]) for (id, tensor) in zip(trial_ids, tensor_list)],
        axis=0,
    )
    tensor = np.concatenate(tensor_list, axis=0)
    (
        train_inds,
        valid_inds,
        train_trial_id,
        valid_trial_id,
        train_data,
        valid_data,
    ) = train_test_split(
        np.arange(tensor.shape[0]), trial_id_map, tensor, test_size=0.2
    )

    with h5py.File(filename, "a") as hf:
        hf.create_dataset("train_data", data=train_data)
        hf.create_dataset("valid_data", data=valid_data)
        hf.create_dataset("train_inds", data=train_inds)
        hf.create_dataset("valid_inds", data=valid_inds)
        hf.create_dataset("train_trial_id", data=train_trial_id)
        hf.create_dataset("valid_trial_id", data=valid_trial_id)


def merge_tensor_into_trials(tensor, trial_ids, overlap):
    """
    Returns a list of arrays, each element corresponding to the signal in a given trial.

    Arguments:
        - tensor (np.ndarray): tensor of neural data (shape: (num_windows, window_len, num_units))
        - trial_ids (np.ndarray): array of trial ids for each tensor
        - overlap (int): number of overlap bins between windows

    Returns:
        - (dict of np.ndarray): dict of rate arrays, one for each trial, with key corresponding to trial id
    """

    return {
        id: merge_chops(tensor[trial_ids == id, :, :], overlap=overlap)
        for id in np.unique(trial_ids)
    }


def add_lfads_rates(
    trial_data,
    lfads_chopped_rates,
    chopped_trial_ids,
    overlap,
    new_sig_name="lfads_rates",
):
    """
    Adds lfads rates to trial_data
    
    Arguments:
        - trial_data (pd.DataFrame): DataFrame in form of PyalData
        - lfads_trial_rates (dict of np.ndarray): dict of rate arrays, one for each trial, with key corresponding to trial id
        - chopped_trial_ids (np.ndarray): array of trial ids for each tensor
        - overlap (int): number of overlap bins between windows
        - new_sig_name (string): name of new signal to add to trial_data
        
    Returns:
        - trial_data (pd.DataFrame): DataFrame with lfads rates added
    """

    trial_data[new_sig_name] = [spikes for spikes in trial_data["M1_spikes"]]

    for frame_ind, frame_trial_id in trial_data["trial_id"].iteritems():
        assert (
            frame_trial_id in chopped_trial_ids
        ), "Trial id {} not found in chopped data".format(frame_trial_id)

        rates = merge_chops(
            lfads_chopped_rates[chopped_trial_ids == frame_trial_id, :, :],
            overlap=overlap,
            orig_len=trial_data.loc[frame_ind, "M1_spikes"].shape[0],
        )

        trial_data.at[frame_ind, new_sig_name] = rates

    return trial_data

