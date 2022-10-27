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

def crystallize_dataframe(td,sig_guide=None):
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

def extract_metaframe(td,metacols=['trial_id']):
    '''
    Extracts a metaframe from a trial dataframe.

    Arguments:
        - td (pd.DataFrame): dataframe in form of PyalData
        - metacols (list of str): columns to include in the metaframe
            Note: if trial_id is not in metacols, it will be added

    Returns:
        - (pd.DataFrame): metaframe with hierarchical index on both axes:
            axis 0: trial id
            axis 1: column name
    '''
    if 'trial_id' not in metacols:
        metacols.insert(0,'trial_id')

    meta_df =  pd.concat([td[col] for col in metacols], axis=1, keys=metacols).set_index('trial_id')
    #meta_df.columns = pd.MultiIndex.from_product([['meta'],meta_df.columns])
    meta_df.columns = pd.MultiIndex.from_tuples(list(zip(meta_df.columns,meta_df.columns)))
    return meta_df

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
