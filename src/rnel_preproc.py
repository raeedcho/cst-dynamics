import scipy
import mat73
import numpy as np
import pandas as pd

def load_ts_data(
    filename: str,
    cols_to_expose: list = ['Kin'],
    index_cols: list = [
        'session_time',
        'trial',
        'task',
        'state',
        'success',
    ],
    signal_cols: list = [
        'pos',
        'vel',
        'targetpos',
        'targetvel',
        'Motor',
        'Sensory',
    ],
    cols_to_explode: list = ['trial_time',],
    single_cols: list = [],
    variable_name: str='TS',
    bin_size: float=20e-3,
):
    try:
        mat = scipy.io.loadmat(
            filename,
            simplify_cells=True,
        )
    except NotImplementedError:
        mat = mat73.loadmat(filename,verbose=False)

    return (
        pd.DataFrame(mat[variable_name])
        .pipe(name_states)
        .pipe(extract_signals,cols_to_expose)
        .pipe(condition_signals,signal_cols)
        .assign(session_time = lambda df: get_session_times(df,bin_size))
        [index_cols+signal_cols]
        .explode(['session_time','state']+signal_cols)
        .astype({'trial': int})
        .set_index(index_cols)
        .pipe(crystallize_dataframe)
    )

def name_states(TS):
    def name_states_row(row):
        row_copy = row.copy()
        if type(row['state_labels'][-1]) is list:
            state_labels = np.array([label[0] for label in row['state_labels']])
        else:
            state_labels = row['state_labels']
        row_copy['state'] = state_labels[(row['state']-1).astype('uint8')]
        return row_copy

    def condition_state_labels(label_list):
        def state_map(label):
            if (
                (type(label) is not str and type(label) is not np.str_)
                or label==''
                or label=='FSafe1'
                or label=='FailSafe1'
            ):
                return 'intertrial'
            else:
                return label.lower()
            
        return np.array(list(map(state_map,label_list)))
        
    return (
        TS
        .apply(name_states_row,axis=1)
        .drop(columns='state_labels')
        .assign(
            state = lambda df: df['state'].map(condition_state_labels),
        )
    )

def extract_signals(TS,cols_to_expose):
    def expose_single_col(s: pd.Series):
        return (
            pd.DataFrame(list(s),s.index)
            .rename(columns=lambda subcol: f'{subcol}'.lower())
        )
    
    return (
        TS
        .assign(**pd.concat(
            [expose_single_col(TS[col]) for col in cols_to_expose],
            axis=1)
        )
    )

def condition_signals(TS,signal_cols):
    return (
        TS
        .assign(**{
            col: lambda df,which_col=col: df[which_col].map(np.transpose)
            for col in signal_cols
        })
    )

def get_session_times(TS,bin_size=20e-3):
    return TS['trial_inds'].map(
        lambda x: pd.to_timedelta(
            (x-1)*bin_size,
            unit='s',
        )
    )

def crystallize_dataframe(df):
    '''
    Expose dataframe columns with numpy arrays as hierarchical columns
    '''
    single_cols = [
        col for col in df.columns
        if type(df[col].iloc[0]) is not np.ndarray
    ]
    array_cols = [
        col for col in df.columns
        if type(df[col].iloc[0]) is np.ndarray
    ]
    return (
        pd.concat(
            [df[col].rename(0) for col in single_cols] +
            [
                pd.DataFrame.from_records(
                    df[array_name].values,
                    df[array_name].index,
                ) for array_name in array_cols
            ],
            axis=1,
            keys=single_cols+array_cols,
        )
        [single_cols+array_cols]
        .rename_axis(columns=['signal','channel'])
    )