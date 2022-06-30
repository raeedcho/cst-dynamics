import pyaldata
import pandas as pd


def generate_realtime_epoch_fun(
    start_point_name, end_point_name=None, rel_start_time=0, rel_end_time=0
):
    """
    Return a function that slices a trial around/between time points (noted in real time, not bins)

    Parameters
    ----------
    start_point_name : str
        name of the time point around which the interval starts
    end_point_name : str, optional
        name of the time point around which the interval ends
        if None, the interval is created around start_point_name
    rel_start_time : float, default 0
        when to start extracting relative to the starting time point
    rel_end_time : float, default 0
        when to stop extracting relative to the ending time point (bin is not included)

    Returns
    -------
    epoch_fun : function
        function that can be used to extract the interval from a trial
    """
    if end_point_name is None:
        epoch_fun = lambda trial: pyaldata.slice_around_point(
            trial,
            start_point_name,
            -rel_start_time / trial["bin_size"],
            rel_end_time / trial["bin_size"] - 1,
        )
    else:
        epoch_fun = lambda trial: pyaldata.slice_between_points(
            trial,
            start_point_name,
            end_point_name,
            -rel_start_time / trial["bin_size"],
            rel_end_time / trial["bin_size"] - 1,
        )

    return epoch_fun

def crystallize_dataframe(td,sigs=['M1_rates','lfads_rates','hand_vel']):
    '''
    Transforms a pyaldata-style dataframe into a normal one, where each row
    is a time point in a trial. This is useful for some manipulations,
    especially those that involve melting the dataframe.

    Arguments:
        - td (pd.DataFrame): dataframe in form of PyalData
        - cols (list of str): columns to include in the crystallized dataframe

    Returns:
        - (pd.DataFrame): crystallized dataframe with hierarchical index on both axes:
            axis 0: trial id, time bin in trial
            axis 1: signal name, signal dimension
    '''
    # TODO: check that signals are in the dataframe and are valid signals

    df = pd.concat(
        [
            pd.concat([pd.DataFrame(trial[sig]) for sig in sigs], axis=1, keys=sigs) 
            for _,trial in td.iterrows()
        ],
        axis=0,
        keys=td['trial_id'],
    )
    df.index.rename('Time bin',level=1,inplace=True)
    return df