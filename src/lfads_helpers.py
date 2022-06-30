import numpy as np
import pyaldata
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from lfads_tf2.utils import chop_data, merge_chops, load_data, load_posterior_averages
from . import data


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
        hf.create_dataset("train_data", data=train_data, compression="gzip")
        hf.create_dataset("valid_data", data=valid_data, compression="gzip")
        hf.create_dataset("train_inds", data=train_inds, compression="gzip")
        hf.create_dataset("valid_inds", data=valid_inds, compression="gzip")
        hf.create_dataset("train_trial_id", data=train_trial_id, compression="gzip")
        hf.create_dataset("valid_trial_id", data=valid_trial_id, compression="gzip")


def merge_tensor_into_trials(tensor, trial_ids, overlap):
    """
    Returns a list of arrays, each element corresponding to the signal in a given trial.

    Arguments:
        - tensor (np.ndarray): tensor of neural data (shape: (num_windows, window_len, num_units))
        - trial_ids (np.ndarray): array of trial ids for each tensor (stored in hdf5 training file)
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

@pyaldata.copy_td
def prep_data_with_lfads(td, task, lfads_params):
    td_task = td.groupby('task').get_group(task).copy()

    td_task = pyaldata.combine_time_bins(td_task,int(lfads_params['bin_size']/td_task['bin_size'].values[0]))

    trial_ids = load_data(
        '../data/pre-lfads/', 
        prefix=f'Earl_20190716_{task}', 
        signal='trial_id',
        merge_tv=True
    )[0].astype(int)

    post_data = load_posterior_averages(f'../results/autolfads/Earl_20190716/autolfads_with_jitter/{task.lower()}', merge_tv=True)
    td_task = add_lfads_rates(
        td_task,
        post_data.rates/lfads_params['bin_size'],
        chopped_trial_ids=trial_ids,
        overlap=lfads_params['overlap'],
        new_sig_name='lfads_rates',
    )
    td_task = add_lfads_rates(
        td_task,
        post_data.gen_inputs/lfads_params['bin_size'],
        chopped_trial_ids=trial_ids,
        overlap=lfads_params['overlap'],
        new_sig_name='lfads_inputs',
    )

    td_task = data.trim_nans(td_task,ref_signals=['rel_hand_pos','lfads_rates'])
    td_task = data.fill_kinematic_signals(td_task)

    td_task['M1_rates'] = [
        pyaldata.smooth_data(
            spikes/bin_size,
            dt=bin_size,
            std=0.05,
            backend='convolve',
        ) for spikes,bin_size in zip(td_task['M1_spikes'],td_task['bin_size'])
    ]

    M1_pca_model = PCA()
    td_task = pyaldata.dim_reduce(td_task,M1_pca_model,'M1_rates','M1_pca')
    
    lfads_pca_model = PCA()
    td_task = pyaldata.dim_reduce(td_task,lfads_pca_model,'lfads_rates','lfads_pca')

    return td_task