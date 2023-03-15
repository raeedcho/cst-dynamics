import pyaldata
import pandas as pd
import numpy as np

def format_outfile_name(trial_data,postfix=''):
    '''
    Format a filename for output based on the trial_data structure
    
    Arguments:
        trial_data (DataFrame): PyalData formatted structure of neural/behavioral data
        postfix (str): postfix to add to the filename
        
    Returns:
        (str): formatted filename
    '''
    filename = '{monkey}_{session_date}_{postfix}'.format(
        monkey=trial_data['monkey'].values[0],
        session_date=np.datetime_as_string(trial_data['session_date'].values[0],'D').replace('-',''),
        postfix=postfix
    )

    return filename
    
def angle_between(v1,v2):
    '''
    Calculate the angle between vectors
    '''
    return (180/np.pi)*np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))

@pyaldata.copy_td
def split_trials_by_epoch(trial_data,epoch_dict,epoch_col_name='epoch'):
    '''
    Split trial_data by epochs
    
    Arguments:
        trial_data (DataFrame): PyalData formatted structure of neural/behavioral data
        epoch_dict (dict): dictionary of epoch functions to pass into
            pyaldata.restrict_to_interval, with keys as epoch names
        epoch_col_name (str): name of column to add to trial_data with epoch names
        
    Returns:
        DataFrame: trial_data with entries for each specified epoch of each trial

    TODO: make this work for exploded dataframes
    '''
    td_epoch_list = []
    for epoch_name,epoch_fun in epoch_dict.items():
        temp_td = pyaldata.restrict_to_interval(
            trial_data,
            epoch_fun=epoch_fun,
        )
        temp_td[epoch_col_name] = epoch_name
        td_epoch_list.append(temp_td)
    
    td_epoch = pd.concat(td_epoch_list).reset_index(drop=True)

    return td_epoch

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

def random_array_like(array):
    '''
    Returns an array of the same size as input,
    with the same overall mean and standard deviation
    (assuming a normal distribution)
    
    Arguments:
        array (np.array): array to imitate
        
    Returns:
        np.array: array with same mean and standard deviation
    '''
    rng = np.random.default_rng()
    return rng.standard_normal(array.shape) * np.std(array) + np.mean(array)
