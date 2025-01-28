import pandas as pd
import numpy as np
import pyaldata
from . import data

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
        sig_guide = pyaldata.get_time_varying_fields(td)

    if type(sig_guide) is list:
        sig_guide = {
            signame: (
                np.arange(td[signame].values[0].shape[1])
                if td[signame].values[0].ndim == 2 else np.arange(1)
            )
            for signame in sig_guide
        }

    assert type(sig_guide) is dict, "sig_guide must be a dictionary"

    temp = (
        td
        .pipe(data.add_trial_time,column_name='trial time')
        .set_index(['monkey','session_date','task','result','trial_id'])
        .filter(items=['trial time']+list(sig_guide.keys()))
    )

    df = pd.concat(
        [
            pd.concat(
                [
                    pd.DataFrame(
                        trial[sig],
                        columns=guide,
                        index=trial['trial time'],
                    ).rename_axis(index='trial time')
                    for sig,guide in sig_guide.items()
                ],
                axis=1,
                keys=sig_guide.keys()
            ).rename_axis(columns=['signal','channel']) 
            for _,trial in temp.iterrows()
        ],
        axis=0,
        keys=temp.index,
    )
    return df

def express_crystallize(td,single_cols,array_cols):
    '''
    Expose dataframe columns with numpy arrays as hierarchical columns

    Arguments:
        - td (pd.DataFrame): dataframe in form of PyalData
        - single_cols (list of str): columns to include as single columns
        - array_cols (list of str): columns to include as hierarchical columns

    Returns:
        - (pd.DataFrame): crystallized dataframe with hierarchical index on both axes:
            axis 0: trial id, time bin in trial
            axis 1: signal name, signal dimension
    '''
    return (
        pd.concat(
            [td[col].rename(0) for col in single_cols] +
            [
                pd.DataFrame(
                    data = np.row_stack(td[array_name]),
                    index = td[array_name].index,
                ) for array_name in array_cols
            ],
            axis=1,
            keys=single_cols+array_cols,
        )
        #.assign(**{(col,0): td[col] for col in single_cols})
        [single_cols+array_cols]
        .rename_axis(columns=['signal','channel'])
    )

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
        #.set_index(['monkey','session_date','trial_id'])
        .set_index('trial_id')
        .filter(items=events)
        .rename_axis('event',axis=1)
        .rename(columns=lambda col: col.replace('idx_','').replace('Times','').replace('Time',''))
        .stack()
        .explode()
        .dropna()
        .sort_values()
        .sort_index(level='trial_id',sort_remaining=False)
        .astype(int)
        .apply(lambda x: x*td['bin_size'].iloc[0])
        .pipe(pd.to_timedelta,unit='s')
    )
    # events_df.columns = pd.MultiIndex.from_product([events_df.columns,['event']])
    return events_df
    
def hierarchical_assign(df,assign_dict):
    '''
    Extends pandas.DataFrame.assign to work with hierarchical columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to assign to
    assign_dict : dict of pandas.DataFrame or callable
        dictionary of dataframes to assign to df
    '''
    return (
        df
        .join(
            pd.concat(
                [val(df) if callable(val) else val for val in assign_dict.values()],
                axis=1,
                keys=assign_dict.keys(),
                names=['signal','channel'],
            )
        )
    )

def sample_trials(tf: pd.DataFrame,timecol: str='trial time',**sample_kwargs):
    '''
    Sample trials from a crystallized dataframe

    Arguments:
        - tf (pd.DataFrame): crystallized dataframe
        - timecol (str): name of the time column
        - sample_kwargs: arguments to pass to tf.sample

    Returns:
        - (pd.DataFrame): sampled dataframe
    '''
    return (
        tf
        .unstack(level=timecol)
        .sample(**sample_kwargs)
        .stack(level=timecol)
    )