"""
This module preps a neural tensor from trial data for LFADS training.
"""

#%%
import src.data
import src.lfads_helpers
import yaml
import pyaldata
import pandas as pd

with open("params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)["lfads_prep"]


def prep_and_save_data(td, save_path):
    """
    This function takes a trial data file and writes a data file with
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
        signal="MC_spikes",
        bin_size=params["bin_size"],
        window_len=params["window_len"],
        overlap=params["overlap"],
    )
    src.lfads_helpers.write_tensor_to_hdf(
        tensor_list, trial_ids, save_path,
    )


#%%
def main():
    # trial_data = src.data.load_clean_data("data/trial_data/Earl_20190716_COCST_TD.mat")
    # td_co = trial_data.groupby("task").get_group("CO")
    # prep_and_save_data(td_co, "data/pre-lfads/Earl_20190716_CO_tensors.hdf5")
    # td_cst = trial_data.groupby("task").get_group("CST")
    # prep_and_save_data(td_cst, "data/pre-lfads/Earl_20190716_CST_tensors.hdf5")

    filename = 'data/trial_data/Prez_20220721_RTTCST_TD.mat'
    # filename = 'data/trial_data/Earl_20190716_COCST_TD.mat'
    td = (
        pyaldata.mat2dataframe(
            filename,
            shift_idx_fields=True,
            td_name='trial_data'
        )
        .assign(
            date_time=lambda x: pd.to_datetime(x['date_time']),
            session_date=lambda x: pd.DatetimeIndex(x['date_time']).normalize()
        )
        .query('task=="RTT" | task=="CST"')
        .pipe(src.data.remove_aborts, verbose=params['verbose'])
        .pipe(src.data.remove_artifact_trials, verbose=params['verbose'])
        .pipe(
            src.data.filter_unit_guides,
            filter_func=lambda guide: guide[:,1] >= (0 if params['keep_unsorted'] else 1)
        )
        .pipe(src.data.remove_correlated_units,verbose=params['verbose'])
        .pipe(
            src.data.remove_all_low_firing_neurons,
            threshold=0.1,
            divide_by_bin_size=True,
            verbose=params['verbose']
        )
    )
    prep_and_save_data(td, "data/pre-lfads/Prez_20220721_RTTCST_tensors.hdf5")


if __name__=='__main__':
    main()
# %%
