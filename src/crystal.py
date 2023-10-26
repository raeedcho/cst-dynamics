import pandas as pd
import pyaldata

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

def extract_trial_event_times(td,events=None):
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
        events = {col for col in td.columns if col.startswith('idx_')}

    events_df = (
        td
        .set_index(['monkey','session_date','trial_id'])
        .filter(items=events)
        .rename_axis('event',axis=1)
        .rename(columns=lambda col: col.replace('idx_','').replace('Times','').replace('Time',''))
        .stack()
        .explode()
        .dropna()
        .sort_values()
        .sort_index(level='trial_id',sort_remaining=False)
    ) * td['bin_size'].iloc[0]
    # events_df.columns = pd.MultiIndex.from_product([events_df.columns,['event']])
    return events_df