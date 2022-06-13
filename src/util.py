import pyaldata


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
