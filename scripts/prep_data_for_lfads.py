"""
This module preps a neural tensor from trial data for LFADS training.
"""

#%%
import src.data
import src.lfads_helpers
import yaml

with open("params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)


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
        **params['chop_merge'],
    )
    src.lfads_helpers.write_tensor_to_hdf(
        tensor_list, trial_ids, save_path,
    )


#%%
def main():
    filename = 'data/trial_data/Prez_20220721_RTTCST_TD.mat'
    td = (
        src.data.preload_data(filename,**params['preload'])
        .query('task=="RTT" | task=="CST"')
    )
    prep_and_save_data(td, "data/pre-lfads/Prez_20220721_RTTCST_tensors.hdf5")

if __name__=='__main__':
    main()