import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from datetime import datetime
import re
import fractions

def convertSMILEtoTD(smile_data, num_random_targs=8, array_name='MC', fill_err=False):
    if params is None:
        params = {}

    params['fill_err'] = False
    smile_data, miss4 = errorCursorSaveFix(smile_data)

    td_list = []
    for trial in smile_data:
        td = {
            'monkey': np.nan,
            'task': np.nan,
            'date_time': np.nan,
            'trial_id': np.nan,
            'result': np.nan,
            'bin_size': np.nan,
            'lambda': np.nan,
            'ct_location': np.nan,
            'rt_locations': np.nan,
            'idx_startTime': np.nan,
            'idx_ctHoldTime': np.nan,
            'idx_pretaskHoldTime': np.nan,
            'idx_goCueTime': np.nan,
            'idx_rtgoCueTimes': np.nan,
            'idx_rtHoldTimes': np.nan,
            'idx_cstStartTime': np.nan,
            'idx_cstEndTime': np.nan,
            'idx_rewardTime': np.nan,
            'idx_failTime': np.nan,
            'idx_endTime': np.nan
        }

        td = parse_smile_meta(td, trial, params)
        td = parse_smile_events(td, trial, params)
        td = parse_smile_behavior(td, trial, params)
        td = parse_smile_eye_data(td, trial, params)
        td = parse_smile_spikes(td, trial, unit_guide, params)

        arrays = split_unit_guide_rows.keys()
        for array_name in arrays:
            array_unit_guide_rows = split_unit_guide_rows[array_name]
            td[f'{array_name}_unit_guide'] = unit_guide[array_unit_guide_rows, :]

            full_spikes = td[f'{array_name}_spikes']
            td[f'{array_name}_spikes'] = full_spikes[:, array_unit_guide_rows]

        td.pop('timevec', None)
        td_list.append(td)

    num_fields = [len(td.keys()) for td in td_list]
    if len(set(num_fields)) > 1:
        print('Warning: something has gone wrong')

    trial_data = td_list
    return trial_data

def compose_session_frame(smile_data, bin_size: str='10ms', **kwargs) -> pd.DataFrame:
    # meta
    meta = get_smile_meta(smile_data)
    states = concat_trial_func_results(get_trial_states, smile_data, bin_size=bin_size)
    hand_pos = concat_trial_func_results(
        get_trial_hand_data,
        smile_data,
        final_sampling_rate=1/pd.to_timedelta(bin_size).total_seconds(),
        **kwargs,
    )
    binned_spikes = (
        get_smile_spike_times(smile_data)
        .pipe(bin_spikes, bin_size=bin_size)
        .pipe(collapse_channel_unit_index)
    )
    
    # targets
    return (
        pd.concat(
            [states,hand_pos,binned_spikes],
            axis=1,
            join='inner',
            keys=['state','hand position','motor cortex'],
            names=['channel','signal'],
        )
        .reset_index(level='time')
        .assign(**meta)
        .set_index('time',append=True)
        [['monkey','date_time','task','result','state','hand position','motor cortex']]
    )

def concat_trial_func_results(trial_func, smile_data: list, **func_kwargs) -> pd.DataFrame:
    return pd.concat(
        [trial_func(trial,**func_kwargs) for trial in smile_data],
        axis=0,
        keys=[get_trial_id(trial) for trial in smile_data],
        names=['trial_id'],
    )

def get_array_channels(monkey_name: str) -> dict:
    if monkey_name == 'Prez':
        array_channels = {
            'M1': np.arange(33, 96),
            'PMd': np.concatenate([np.arange(1,33), np.arange(96, 129)]),
        }
    elif monkey_name == 'Dwight':
        array_channels = {
            'M1': np.concatenate([np.arange(1,33), np.arange(96, 129)]),
            'PMd': np.arange(33, 96),
        }

    return array_channels

def get_trial_state_table(smile_trial: dict) -> pd.DataFrame:
    # return pd.from_dict(
    #     smile_trial['Parameters']['StateTable'],
    # )
    pass

def parse_smile_meta(td, trial, meta={}, num_random_targs=8):
    ct_reach_state = next(state for state in trial['Parameters']['StateTable'] if state['stateName'] == 'Reach to Center')
    center_target_idx = [i for i, name in enumerate(ct_reach_state['StateTargets']['names']) if name in ['starttarget', 'start']]
    td['ct_location'] = ct_reach_state['StateTargets']['location'][center_target_idx[0]]
    td['ct_location'][1] = -td['ct_location'][1]

    if trial_name.startswith('RandomTargetTask_20220630'):
        td['rt_locations'] = np.zeros((num_random_targs, 3))
        for targetnum in range(num_random_targs):
            targ_reach_state = next(state for state in trial['Parameters']['StateTable'] if state['stateName'] == f'Reach to Target {targetnum}')
            targ_idx = next(i for i, name in enumerate(targ_reach_state['StateTargets']['names']) if name == f'randomtarg{targetnum}')
            td['rt_locations'][targetnum] = targ_reach_state['StateTargets']['location'][targ_idx]
        td['rt_locations'][:, 1] = -td['rt_locations'][:, 1]

    if 'CST' in trial_name:
        td['lambda'] = trial['Parameters']['ForceParameters']['initialLambda']

    for key, value in meta.items():
        td[key] = value

    return td

def get_trial_targets(smile_trial: dict) -> pd.DataFrame:
    pass

def get_trial_id(smile_trial: dict) -> int:
    return int(smile_trial['Overview']['trialNumber'].replace('Trial',''))

def get_trial_datetime_str(smile_trial: dict) -> str:
    return (
        datetime.strptime(
            str(int(smile_trial['Overview']['date'])),
            '%Y%m%d%H%M%S',
        ).strftime('%Y-%m-%d %H:%M:%S')
    )

def get_trial_result(smile_trial: dict) -> str:
    result_code = smile_trial['Overview']['trialStatus']
    if type(result_code) is np.ndarray:
        result_code = result_code.item()

    return (
        {
            0: 'failure',
            1: 'success',
            2: 'abort',
        }[result_code]
    )

def get_trial_task(smile_trial: dict) -> str:
    trial_name = smile_trial['Overview']['trialName']
    if trial_name.startswith('RandomTargetTask'):
        task = 'RTT'
    elif trial_name.startswith('CST'):
        task = 'CST'
    elif trial_name.startswith('CenterOut'):
        task = 'CO'
    elif re.match(r'^R.T.$', trial_name):
        task = 'DCO'
    elif re.match(r'^R.T.C$', trial_name):
        task = 'DCO-catch'
    else:
        task = trial_name

    return task

def get_smile_meta(smile_data: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'monkey': smile_trial['Overview']['subjectName'],
                'date_time': get_trial_datetime_str(smile_trial),
                'task': get_trial_task(smile_trial),
                'result': get_trial_result(smile_trial),
            }
            for smile_trial in smile_data
        ],
        index=pd.Index(
            [get_trial_id(smile_trial) for smile_trial in smile_data],
            name='trial_id',
        )

    )

def get_trial_events(smile_trial: dict) -> pd.DataFrame:
    def get_state_name(state_id):
        if state_id == -1:
            return 'end'
        else:
            return smile_trial['Parameters']['StateTable'][state_id-1]['stateName']

    return (
        pd.DataFrame(
            [
                {
                    'event': get_state_name(state_id),
                    'time': pd.to_timedelta(event_frame-1, unit='ms'),
                }
                for [state_id,event_frame] in smile_trial['TrialData']['stateTransitions'].T
            ],
        )
        .set_index('event')
    )

def get_trial_states(smile_trial: dict, bin_size: str='10ms') -> pd.Series:
    return (
        get_trial_events(smile_trial)
        .reset_index(level='event')
        .set_index('time')
        .resample(bin_size)
        .ffill()
        .squeeze()
        .rename('')
    )

# Spikes
def get_trial_spike_times(trial: dict, keep_sorted_only=True) -> pd.DataFrame:
    trial_spikes = (
        pd.DataFrame(
            trial['TrialData']['TDT']['snippetInfo'].T,
            columns=['channel', 'unit', 'frame'],
        )
        .assign(**{
            'timestamp': lambda x: pd.to_timedelta(x['frame']-1, unit='ms'),
        })
        .drop(columns='frame')
        .rename_axis('snippet_id', axis=0)
    )

    if keep_sorted_only:
        trial_spikes = trial_spikes.loc[lambda x: (x['unit']>0) & (x['unit']<31)]
    
    return trial_spikes



def get_trial_waveforms(trial: dict) -> pd.DataFrame:
    waveforms = trial['TrialData']['TDT']['snippetWaveforms'].T
    return (
        pd.DataFrame(
            waveforms,
            columns=pd.RangeIndex(start=0, stop=waveforms.shape[1], name='snippet frame'),
        )
        .rename_axis('snippet_id', axis=0)
        .assign(**get_trial_spike_times(trial,keep_sorted_only=False)[['channel','unit']])
        .set_index(['channel','unit'],append=True)
    )

def get_smile_spike_times(smile_data: list, keep_sorted_only=True) -> pd.DataFrame:
    return concat_trial_func_results(get_trial_spike_times, smile_data, keep_sorted_only=keep_sorted_only)

def get_spike_waveforms(smile_data: list) -> pd.DataFrame:
    return concat_trial_func_results(get_trial_waveforms, smile_data)

def bin_spikes(spike_times: pd.DataFrame, bin_size: str='10ms') -> pd.DataFrame:
    # may need to adjust this to handle cases where there are no spikes at time 0
    # possibly using .asfreq() before slicing to ensure that all timepoints are present
    spike_counts = (
        spike_times
        .assign(spikes=1)
        .set_index(['channel','unit','timestamp'],append=True)
        .reset_index(level='snippet_id',drop=True)
        .squeeze()
        .unstack(level=['channel','unit'],fill_value=0)
        .sort_index(level=['trial_id','timestamp'],axis=0)
        .sort_index(level=['channel','unit'],axis=1)
        .loc[(slice(None),slice('0s',None)),:]
        .groupby('trial_id')
        .resample(bin_size,level='timestamp')
        .sum()
        .rename_axis(index={'timestamp':'time'})
    )

    return spike_counts

def collapse_channel_unit_index(binned_spikes: pd.DataFrame) -> pd.DataFrame:
    assert binned_spikes.columns.names == ['channel','unit'], "Columns must be MultiIndex with channel and unit levels."

    signal_ids = binned_spikes.columns.to_series().apply(lambda x: f'ch{x[0]}u{x[1]}')
    return pd.DataFrame(
        binned_spikes.values,
        index=binned_spikes.index,
        columns=pd.Index(signal_ids,name='signal'),
    )

# Phasespace data
def get_trial_hand_data(
        smile_trial: dict,
        resample_window: tuple=('kaiser',20.0),
        final_sampling_rate: float=1000,
        reference_loc:np.array = None,
        **kwargs,
    ) -> pd.DataFrame:

    if reference_loc is None:
        reference_loc = np.array([0,0,0])

    phasespace_data = smile_trial['TrialData']['Marker']['rawPositions']
    if phasespace_data.shape[0] == 0:
        return pd.DataFrame()

    phasespace_freq = smile_trial['TrialData']['Marker']['frequency']
    marker_position = (phasespace_data[:,1:4] - reference_loc)# * [1,-1,1] # flip y-axis for data collected in rig 1 before 2023-10-01
    framevec = (phasespace_data[:,4]).astype(int)
    full_framevec = np.arange(framevec[0], framevec[-1]+1)
    final_timevec = pd.timedelta_range(
        start=0,
        end=convert_phasespace_frame_to_time(full_framevec[-1], smile_trial),
        freq=pd.to_timedelta(1/final_sampling_rate, unit='s'),
        name='time',
    )

    marker_pos_interp = (
        pd.DataFrame(
            marker_position,
            columns=pd.Index(['x','y','z'],name='signal'),
            index=pd.Index(framevec,name='phasespace_frame')
        )
        .reindex(full_framevec)
        .interpolate(method='linear')
        .reset_index()
        .assign(
            time=lambda x: convert_phasespace_frame_to_time(x['phasespace_frame'], smile_trial),
        )
        .set_index('time')
        .drop(columns='phasespace_frame')
        .pipe(sig_resample, final_sampling_rate, old_sampling_rate=phasespace_freq, window=resample_window)
        .pipe(interpolating_reindex, final_timevec)
    )

    return marker_pos_interp

def convert_phasespace_frame_to_time(framevec, smile_trial):
    phasespace_sync_frame = smile_trial['TrialData']['Marker']['SyncParameters']['phasespaceFrame']
    phasespace_sync_time = pd.to_timedelta(smile_trial['TrialData']['Marker']['SyncParameters']['startTime'],unit='ms')
    phasespace_freq = smile_trial['TrialData']['Marker']['frequency']
    return pd.to_timedelta((framevec-phasespace_sync_frame)/phasespace_freq,unit='s') + phasespace_sync_time

def get_trial_eye_data(smile_trial: dict) -> pd.DataFrame:
    pass

def sig_resample(df: pd.DataFrame, new_sampling_rate: float, old_sampling_rate: float=None, **kwargs)->pd.DataFrame:
    assert type(df.index) is pd.TimedeltaIndex, "Index must be a TimedeltaIndex."

    if old_sampling_rate is None:
        old_timevec_period = (
            df.index
            .diff()
            .value_counts()
            .idxmax()
            .total_seconds()
        )
        old_sampling_rate = 1/old_timevec_period

    resample_factor = fractions.Fraction.from_float(new_sampling_rate / old_sampling_rate).limit_denominator()
    new_signal = resample_poly(
        df.values,
        resample_factor.numerator,
        resample_factor.denominator,
        axis=0,
        padtype='line',
        **kwargs,
    )
    new_timevec = pd.timedelta_range(
        start=df.index[0],
        periods=new_signal.shape[0],
        freq=pd.to_timedelta(1/new_sampling_rate, unit='s'),
        name='time',
    )

    return pd.DataFrame(
        index=new_timevec,
        data=new_signal,
        columns=df.columns,
    )

def multicol_interp(x, xp, fp, **kwargs):
    assert xp.shape[0] == fp.shape[0], "xp and fp must have the same number of rows."
    assert x.ndim == 1, "x must be 1D."
    assert xp.ndim == 1, "xp must be 1D."
    assert fp.ndim == 2, "fp must be 2D."

    return np.column_stack([np.interp(x, xp, fp[:,i], **kwargs) for i in range(fp.shape[1])])

def interpolating_reindex(df, new_index):
    assert type(new_index) is pd.TimedeltaIndex, "new_index must be a pandas Index."

    return pd.DataFrame(
        index=new_index,
        data=multicol_interp(new_index, df.index, df.values),
        columns=df.columns,
    )

# CST error cursor
def get_trial_cst_cursor(smile_trial: dict) -> pd.DataFrame:
    pass