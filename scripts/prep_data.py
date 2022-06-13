"""
This module preps a neural tensor from trial data for LFADS training.
"""

import src.data
import src.lfads_helpers
import yaml

with open("params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)["lfads_prep"]


def prep_and_save_data(td, save_path):
    """
    This function takes a trial data file and returns a trial data file with
    the neural tensor and lfads rates added to it.

    Parameters
    ----------
    td : pandas.DataFrame
        Trial data file.
    save_path : str
        Path to save the trial data file.
    """
    tensor_list, trial_ids = src.lfads_helpers.prep_neural_tensors(
        td,
        signal="M1_spikes",
        bin_size=params["bin_size"],
        window_len=params["window_len"],
        overlap=params["overlap"],
    )
    src.lfads_helpers.write_tensor_to_hdf(
        tensor_list, trial_ids, save_path,
    )


trial_data = src.data.load_clean_data("data/trial_data/Earl_20190716_COCST_TD.mat")

td_co = trial_data.groupby("task").get_group("CO")
prep_and_save_data(td_co, "data/pre-lfads/Earl_20190716_CO_tensors.hdf5")
td_cst = trial_data.groupby("task").get_group("CST")
prep_and_save_data(td_cst, "data/pre-lfads/Earl_20190716_CST_tensors.hdf5")

