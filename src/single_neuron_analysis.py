import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pyaldata
from . import util
from sklearn.linear_model import PoissonRegressor

def get_condition_neural_averages(trial_data,signal='',cond_col=None):
    '''
    Takes trial data as input and returns a table of average firing rates per neuron,
    separated by specified condition column

    Arguments:
        trial_data (DataFrame): PyalData formatted structure of neural/behavioral data
        signal (str): name of signal to use for calculating neural averages
            (Note: signal must end in '_spikes' or '_rate')
        cond_col (str): name of column to use for grouping trials by condition

    Returns:
        DataFrame: table of average firing rates per neuron, with columns:
            ['monkey','session_date','task','epoch','array','chan_id','unit_id','average_rate']

    TODO: combine this with get_task_epoch_neural_averages by passing in cond_col as a list
        (possibly using pandas groupby functionality)
    '''
    array = util.get_array_from_signal(signal)
    unit_guide = trial_data.loc[trial_data.index[0], array+'_unit_guide']

    conditions = trial_data[cond_col].unique()

    fr_table_list = []
    for cond in conditions:
        temp_td = trial_data.loc[trial_data[cond_col] == cond, :].copy()
        avg_fr = pyaldata.get_average_firing_rates(temp_td, array+'_spikes', divide_by_bin_size=True)
        fr_table_list.append(pd.DataFrame({
            'monkey': temp_td['monkey'].values[0],
            'session_date': temp_td['session_date'].values[0],
            'array': array,
            'chan_id': unit_guide[:,0],
            'unit_id': unit_guide[:,1],
            cond_col: cond,
            'average_rate': avg_fr
        }))

    fr_table = pd.concat(fr_table_list)
    fr_table.reset_index(drop=True,inplace=True)

    return fr_table

def get_task_epoch_neural_averages(trial_data,signal='',hold_start=-0.4):
    '''
    Takes trial data as input and returns a table of average firing rates per neuron,
    separated by array, task (CO or RTT), and epoch (hold or move)

    Arguments:
        trial_data (DataFrame): PyalData formatted structure of neural/behavioral data
        signal (str): name of signal to use for calculating neural averages
            (Note: signal must end in '_spikes' or '_rate')
        hold_start (float): time (in sec) relative go cue to consider the hold time for CO and CST trials
            (default: -0.4)

    Returns:
        DataFrame: table of average firing rates per neuron, with columns:
            ['monkey','session_date','task','epoch','array','chan_id','unit_id','average_rate']
    '''
    array = util.get_array_from_signal(signal)
    unit_guide = trial_data.loc[trial_data.index[0], array+'_unit_guide']

    tasks = ['CO', 'CST']
    epochs = ['hold', 'move']

    fr_table_list = []
    for task in tasks:
        for epoch in epochs:
            temp_td = trial_data.loc[trial_data['task']==task,:].copy()
            if epoch == 'hold':
                temp_td = pyaldata.restrict_to_interval(
                    temp_td,
                    start_point_name='idx_goCueTime',
                    rel_start=hold_start//temp_td['bin_size'].values[0],
                    rel_end=0,
                    reset_index=False
                )
            elif epoch == 'move' and task == 'CO':
                temp_td = pyaldata.restrict_to_interval(
                    temp_td,
                    start_point_name='idx_goCueTime',
                    end_point_name='idx_otHoldTime',
                    reset_index=False
                )
            elif epoch == 'move' and task == 'CST':
                temp_td = pyaldata.restrict_to_interval(
                    temp_td,
                    start_point_name='idx_cstStartTime',
                    end_point_name='idx_cstEndTime',
                    reset_index=False
                )

            avg_fr = pyaldata.get_average_firing_rates(temp_td,array+'_spikes',divide_by_bin_size=True)

            fr_table_list.append(pd.DataFrame({
                'monkey': temp_td['monkey'].values[0],
                'session_date': temp_td['session_date'].values[0],
                'array': array,
                'chan_id': unit_guide[:,0],
                'unit_id': unit_guide[:,1],
                'task': task,
                'epoch': epoch,
                'average_rate': avg_fr
            }))

    fr_table = pd.concat(fr_table_list)
    fr_table.reset_index(drop=True,inplace=True)

    return fr_table

def plot_task_epoch_neural_averages(avg_fr_table):
    '''
    Takes a table of average firing rates per neuron, separated by array, task (CO or CST), and epoch (hold or move)
    Note: this function alters the current state of matplotlib by using the pyplot state machine

    Arguments:
        avg_fr_table (DataFrame): table of average firing rates per neuron, with columns:
            ['monkey','session_date','task','epoch','array','chan_id','unit_id','average_rate']

    Returns:
        None
    '''
    max_fr = avg_fr_table['average_rate'].max()
    avg_fr_pivot = avg_fr_table.pivot(index=['chan_id','unit_id'],columns=['task','epoch'],values='average_rate')
    avg_fr_pivot.columns = avg_fr_pivot.columns.to_series().apply(' '.join)
    axs = pd.plotting.scatter_matrix(avg_fr_pivot,figsize=(10,10))
    for colnum,col in enumerate(axs):
        for rownum,ax in enumerate(col):
            plt.setp(ax.yaxis.get_majorticklabels(),'size',18)
            plt.setp(ax.xaxis.get_majorticklabels(),'size',18)
            ax.set_xlim([0, max_fr+5])
            if rownum != colnum:
                ax.plot([0,max_fr+5],[0,max_fr+5],'--k')
                ax.set_ylim([0,max_fr+5])

    plt.suptitle('Task/Epoch average firing rate comparison')

def get_task_neural_lr_selectivity(trial_data,signal=''):
    '''
    Takes trial data as input and returns a table of left/right selectivity per neuron,
    separated by array, task (CO or CST). Left/right selectivity is the coefficient
    relating the x-axis velocity to the neuron's firing rate (units: (spikes/sec)/(m/sec)).
    Essentially, this is the slope of the Poisson regression of the neuron's firing rate
    vs. x-axis velocity.

    Arguments:
        trial_data (DataFrame): PyalData formatted structure of neural/behavioral data
        signal (str): name of signal corresponding to neural data

    Returns:
        DataFrame: table of average tuning curves per neuron, with columns:
            ['monkey','session_date','task','array','chan_id','unit_id','lr_selectivity']
    '''
    array = util.get_array_from_signal(signal)
    unit_guide = trial_data.loc[trial_data.index[0], array+'_unit_guide']
    tasks = ['CO', 'CST']

    lr_selectivity_table_list = []
    for task in tasks:
        temp_td = trial_data.loc[trial_data['task']==task,:].copy()
        temp_td = pyaldata.restrict_to_interval(
            temp_td,
            start_point_name='idx_goCueTime',
            end_point_name='idx_endTime',
            rel_end=-1,
            reset_index=False
        )
        temp_td = pyaldata.combine_time_bins(temp_td,int(0.05//temp_td['bin_size'].values[0]))

        lr_selectivity_table_list.append(pd.DataFrame({
            'monkey': temp_td['monkey'].values[0],
            'session_date': temp_td['session_date'].values[0],
            'task': task,
            'array': array,
            'chan_id': unit_guide[:,0],
            'unit_id': unit_guide[:,1],
            'lr_selectivity': get_lr_selectivity(
                np.row_stack(temp_td['hand_vel'])[:,0][:,np.newaxis],
                np.row_stack(temp_td[signal]),
            ).squeeze()
        }))

    lr_selectivity_table = pd.concat(lr_selectivity_table_list)
    lr_selectivity_table.reset_index(drop=True,inplace=True)

    return lr_selectivity_table

def get_lr_selectivity(hand_vel,neural_data):
    '''
    Takes trial data as input and returns a numpy array of left/right selectivity
    per neuron in trial_data[signal]. Left/right selectivity is the coefficient
    of the Poisson regression relating the x-axis hand velocity to the neuron's firing rate
    (units: (spikes/sec)/(m/sec)).

    Arguments:
        hand_vel (np.array): array of horizontal hand velocities, shape: (n_time x 1)
        neural_data (np.array): neural signal to use for calculating selectivity, shape: (n_time x n_neurons)

    Returns:
        numpy array: array of left/right selectivity per neuron in trial_data[signal]
    '''
    if hand_vel.ndim==1:
        hand_vel=hand_vel[:,np.newaxis]

    lr_sensitivity = np.zeros(neural_data.shape[1])
    for neuronnum in range(neural_data.shape[1]):
        lr_sensitivity_model = PoissonRegressor() # TODO: make type of regression a parameter
        lr_sensitivity_model.fit(hand_vel,neural_data[:,neuronnum])
        lr_sensitivity[neuronnum] = lr_sensitivity_model.coef_[0]

    return lr_sensitivity

def plot_task_neural_lr_selectivity(lr_selectivity_table):
    '''
    Takes a table of left/right selectivity per neuron, separated by array, task (CO or CST)
    Note: this function alters the current state of matplotlib by using the pyplot state machine

    Arguments:
        lr_selectivity_table (DataFrame): table of left/right selectivity per neuron, with columns:
            ['monkey','session_date','task','array','chan_id','unit_id','lr_selectivity']

    Returns:
        None
    '''
    lr_selectivity_pivot = lr_selectivity_table.pivot(index=['chan_id','unit_id'],columns='task',values='lr_selectivity')
    plt.scatter(lr_selectivity_pivot['CO'],lr_selectivity_pivot['CST'])
    plt.plot([-0.02,0.02],[-0.02,0.02],'--k')
    plt.plot([-0.02,0.02],[0,0],'-k')
    plt.plot([0,0],[-0.02,0.02],'-k')
    plt.xlabel('CO sensitivity')
    plt.ylabel('CST sensitivity')
    plt.suptitle('Task neural left/right selectivity (Poisson)')
    sns.despine()