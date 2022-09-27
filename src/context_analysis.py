import pyaldata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import subspace_tools,data,util

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,GroupShuffleSplit

def extract_td_epochs(td):
    '''
    Prepare data for hold-time PCA and LDA, as well as data for smooth hold/move M1 activity
    
    Arguments:
        args (Namespace): Namespace of command-line arguments
        
    Returns:
        td_binned (DataFrame): PyalData formatted structure of neural/behavioral data
        td_smooth (DataFrame): PyalData formatted structure of neural/behavioral data
    '''
    binned_epoch_dict = {
        'ambig_hold': util.generate_realtime_epoch_fun(
            'idx_pretaskHoldTime',
            rel_start_time=-0.3,
        ),
        'hold': util.generate_realtime_epoch_fun(
            'idx_goCueTime',
            rel_start_time=-0.3,
        ),
        'move': util.generate_realtime_epoch_fun(
            'idx_goCueTime',
            rel_start_time=0,
            rel_end_time=0.3,
        ),
    }

    td_binned = (
        td.copy()
        .pipe(util.split_trials_by_epoch,binned_epoch_dict)
        .pipe(data.rebin_data,new_bin_size=0.3)
    )

    spike_fields = [name for name in td.columns.values if name.endswith("_spikes")]
    for field in spike_fields:
        assert td_binned[field].values[0].ndim==1, "Binning didn't work"

    smooth_epoch_dict = {
        'hold_move': util.generate_realtime_epoch_fun(
            'idx_goCueTime',
            rel_start_time=-0.8,
            rel_end_time=0.5,
        ),
        'hold_move_ref_cue': util.generate_realtime_epoch_fun(
            'idx_pretaskHoldTime',
            rel_start_time=-0.3,
            rel_end_time=1.0,
        ),
        'full_trim': util.generate_realtime_epoch_fun(
            'idx_goCueTime',
            rel_start_time=-0.8,
            rel_end_time=5,
        ),
        'full': lambda trial : slice(0,trial['hand_pos'].shape[0]),
    }
    td_smooth = (
        td.copy()
        .pipe(util.split_trials_by_epoch,smooth_epoch_dict)
    )

    td_epochs = pd.concat([td_binned,td_smooth]).reset_index()

    return td_epochs

@pyaldata.copy_td
def apply_models(td,train_epochs=None,test_epochs=None,label_col='task'):
    '''
    Apply PCA and LDA models to hold-time data

    Note: only returns the data of the chosen epochs
    '''
    
    if type(train_epochs)==str:
        train_epochs = [train_epochs]
    if type(test_epochs)==str:
        test_epochs = [test_epochs]

    assert type(train_epochs)==list, "train_epochs must be a list"
    assert type(test_epochs)==list, "test_epochs must be a list"

    td_train = td.loc[td['epoch'].isin(train_epochs),:].copy()
    td_test = td.loc[td['epoch'].isin(test_epochs),:].copy()
    
    beh_lda_model = LinearDiscriminantAnalysis()
    td_train['beh_lda'] = beh_lda_model.fit_transform(
        np.column_stack([
            np.row_stack(td_train['rel_hand_pos'].values),
            np.row_stack(td_train['hand_vel'].values),
        ]),
        td_train[label_col]
    )
    td_train['beh_pred'] = beh_lda_model.predict(
        np.column_stack([
            np.row_stack(td_train['rel_hand_pos'].values),
            np.row_stack(td_train['hand_vel'].values),
        ])
    )
    td_test['beh_lda'] = [beh_lda_model.transform(np.column_stack([pos,vel])) for pos,vel in zip(td_test['rel_hand_pos'],td_test['hand_vel'])]
    td_test['beh_pred'] = [beh_lda_model.predict(np.column_stack([pos,vel])) for pos,vel in zip(td_test['rel_hand_pos'],td_test['hand_vel'])]

    arrays = [name.replace('_rates','') for name in td_train.columns if name.endswith('_rates')]
    lda_pipes={}
    for array in arrays:
        lda_pipes[array] = Pipeline([
            ('pca',PCA(n_components=15)),
            ('lda',LinearDiscriminantAnalysis()),
        ])
        td_train[f'{array}_lda'] = lda_pipes[array].fit_transform(
            np.row_stack(td_train[f'{array}_rates'].values),
            td_train[label_col]
        )
        td_train[f'{array}_pred'] = lda_pipes[array].predict(np.row_stack(td_train[f'{array}_rates']))
        td_test[f'{array}_lda'] = [lda_pipes[array].transform(sig) for sig in td_test[f'{array}_rates']]
        td_test[f'{array}_pred'] = [lda_pipes[array].predict(sig) for sig in td_test[f'{array}_rates']]

    return td_train,td_test,lda_pipes

def plot_separability_dynamics(td,ref_time_col,time_lims=[-0.8,5],ax=None,pred_name='M1_pred'):
    '''
    Plot the dynamics of how well an LDA model separates the two tasks in the smoothed data

    Args:
        td (DataFrame): PyalData formatted structure of neural/behavioral data
        ref_time_col (str): name of column to reference time by (used for aggregation)
            Options are either "Time from go cue (s)" or "Time from pretask hold (s)"
        time_lims (list): limits of time axis (default: [-0.8,5])
        ax (Axes): axes to plot on (default: None--creates new figure)
        pred_name (str): name of column to use for prediction (default: 'M1_pred')

    Returns:
        ax (Axes): axes with plot
    '''
    sep_df_ref = (
        td.copy()
        .pipe(data.add_trial_time,ref_event='idx_goCueTime',column_name='Time from go cue (s)')
        .pipe(data.add_trial_time,ref_event='idx_pretaskHoldTime',column_name='Time from pretask hold (s)')
        .loc[:,['trial_id','task','epoch','Time from go cue (s)','Time from pretask hold (s)',pred_name]]
        .explode(['Time from go cue (s)','Time from pretask hold (s)',pred_name])
        .assign(classify_success=lambda x: x[pred_name]==x['task'])
        .assign(trialtime_hash=lambda x: pd.to_timedelta(x[ref_time_col],'s'))
        .loc[lambda x: (x[ref_time_col]>=time_lims[0]) & (x[ref_time_col]<=time_lims[1]),:]
        .groupby('trialtime_hash')
        .agg(**{
            'Separability': ('classify_success',np.mean),
            ref_time_col: (ref_time_col,np.mean),
        })
    )

    if ax is None:
        ax = plt.gca()

    sns.lineplot(
        ax=ax,
        data=sep_df_ref,
        x=ref_time_col,
        y='Separability',
    )
    ax.plot(time_lims,[0.5,0.5],'k--')
    ax.set_xlim(time_lims)
    ax.set_ylim([0,1])
    sns.despine(ax=ax,trim=True)

    return ax

def plot_any_dim_separability(td,signal='M1_pca',ref_time_col='Time from go cue (s)',time_lims=[-0.8,5],ax=None):
    '''
    Calculates and plots how separable neural states are between tasks. This function
    first runs PCA on neural activity, then for each timepoint, fits an LDA model to
    determine how separable tasks are at that time across trials (aligned by whatever
    reference is provided). The function then plots out the timecourse of this
    separability in the provided Axes.

    Args:
        td (DataFrame): PyalData formatted structure of neural/behavioral data
        signal (str): name of signal column to work on (default: 'M1_pca')
        ref_time_col (str): name of column to reference time by (used for aggregation)
            Options are either "Time from go cue (s)" or "Time from pretask hold (s)"
        time_lims (list): limits of time axis (default: [-0.8,5])
        ax (Axes): axes to plot on (default: None--creates new figure)

    Returns:
        ax (Axes): axes with plot
    '''

    sep_df_ref = (
        td.copy()
        .pipe(data.add_trial_time,ref_event='idx_goCueTime',column_name='Time from go cue (s)')
        .pipe(data.add_trial_time,ref_event='idx_pretaskHoldTime',column_name='Time from pretask hold (s)')
        .loc[:,['trial_id','task','epoch','Time from go cue (s)','Time from pretask hold (s)',signal]]
        .explode(['Time from go cue (s)','Time from pretask hold (s)',signal])
        .loc[lambda x: (x[ref_time_col]>=time_lims[0]) & (x[ref_time_col]<=time_lims[1]),:]
        .assign(trialtime_hash=lambda x: pd.to_timedelta(x[ref_time_col],'s'))
        .groupby('trialtime_hash')
        .apply(lambda x: pd.Series({
            'Separability': np.mean(cross_val_score(
                LinearDiscriminantAnalysis(),
                np.row_stack(x[signal]),
                x['task'],
                cv=5,
            )),
            ref_time_col: np.mean(x[ref_time_col]),
        }))
    )

    if ax is None:
        ax = plt.gca()

    sns.lineplot(
        ax=ax,
        data=sep_df_ref,
        x=ref_time_col,
        y='Separability',
    )
    ax.plot(time_lims,[0.5,0.5],'k--')
    ax.set_xlim(time_lims)
    ax.set_ylim([0,1])
    sns.despine(ax=ax,trim=True)

    return ax

def get_train_test_separability(td,signal='M1_pca',ref_time_col='Time from go cue (s)',time_lims=[-0.8,5],ax=None):
    '''
    Get the separability traces of neural activity based on task, where each trace is the separability
    over the whole trial along a single dimension found using a given timepoint.

    Args:
        td (DataFrame): PyalData formatted structure of neural/behavioral data
        signal (str): name of signal column to work on (default: 'M1_pca')
        ref_time_col (str): name of column to reference time by (used for aggregation)
            Options are either "Time from go cue (s)" or "Time from pretask hold (s)"
        time_lims (list): limits of time axis (default: [-0.8,5])
        ax (Axes): axes to plot on (default: None--creates new figure)

    Returns:
        (pd.Series): Series of separability trace indexed by training time and testing time
        (pd.Series): Series of LDA axis coefficients indexed by training time
    '''
    # set up function to train at a given time in training set and predict for all times in training set
    
    sep_df_ref = (
        td.copy()
        .pipe(data.add_trial_time,ref_event='idx_goCueTime',column_name='Time from go cue (s)')
        .pipe(data.add_trial_time,ref_event='idx_pretaskHoldTime',column_name='Time from pretask hold (s)')
        .loc[:,['trial_id','task','epoch','Time from go cue (s)','Time from pretask hold (s)',signal]]
        .explode(['Time from go cue (s)','Time from pretask hold (s)',signal])
        .loc[lambda x: (x[ref_time_col]>=time_lims[0]) & (x[ref_time_col]<=time_lims[1]),:]
        .assign(trialtime_hash=lambda x: pd.to_timedelta(x[ref_time_col],'s'))
    )
    train_test_index = pd.MultiIndex.from_product(
        [sep_df_ref['trialtime_hash'].unique(),sep_df_ref['trialtime_hash'].unique()],
        names=['train_time','test_time'],
    )

    gss = GroupShuffleSplit(n_splits=1,test_size=0.25)
    for train_idx,test_idx in gss.split(sep_df_ref,groups=sep_df_ref['trial_id']):
        df_train = sep_df_ref.iloc[train_idx]
        df_test = sep_df_ref.iloc[test_idx]
        
        train_times = []
        test_seps = []
        train_lda_coefs = []
        for train_time,train_rows in df_train.groupby('trialtime_hash'):
            lda_model = LinearDiscriminantAnalysis()
            lda_model.fit(np.row_stack(train_rows[signal]),train_rows['task'])
            train_lda_coefs.append(lda_model.coef_.squeeze())

            train_times.append(train_time)
            test_seps.append(
                df_test
                .groupby('trialtime_hash')
                .apply(lambda x: lda_model.score(
                        np.row_stack(x[signal]),
                        x['task'],
                    ),
                )
            )

        sep_series = pd.concat(test_seps,keys=train_times,names=['train_time','test_time']).rename('Separability')
        lda_coefs = pd.Series(train_lda_coefs,index=train_times)

    return sep_series,lda_coefs

def plot_hold_pca(td,array_name='M1',label_col='task',hue_order=['CO','CST']):
    '''
    Plot the M1 neural population activity for trials separated by label (e.g. task)

    Arguments:
        td (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        pca_fig (Figure): Figure of PCA plot
    '''
    assert type(label_col)==str, "label_col must be a string"

    pca_fig,pca_ax = plt.subplots(1,1,figsize=(6,4))
    sns.scatterplot(
        ax=pca_ax,
        data=td,
        x=np.row_stack(td[f'{array_name}_pca'].values)[:,0],
        y=np.row_stack(td[f'{array_name}_pca'].values)[:,1],
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    pca_ax.set_ylabel(f'{array_name} PC2')
    pca_ax.set_xlabel(f'{array_name} PC1')
    sns.despine(ax=pca_ax,trim=True)

    return pca_fig

def plot_M1_lda_traces(td_smooth,ref_event='idx_goCueTime',label_col='task',label_colors={'CO':'r','CST':'b'}):
    '''
    Plot out M1 activity through hold period and first part of trial
    projected through LDA axis fit on average hold activity to separate
    tasks.

    Arguments:
        td_smooth (DataFrame): PyalData formatted structure of neural/behavioral data
        label_col (str): Column name of label (e.g. task)

    Returns:
        lda_fig (Figure): Figure of LDA plot
    '''
    assert type(label_col)==str, "label_col must be a string"

    lda_fig,lda_ax = plt.subplots(1,1,figsize=(6,4))
    for _,trial in data.add_trial_time(td_smooth,ref_event=ref_event).iterrows():
        lda_ax.plot(
            trial['trialtime'],
            trial['M1_lda'][:,0],
            c=label_colors[trial[label_col]],
            alpha=0.2,
        )
    lda_ax.plot(
        [0,0],
        lda_ax.get_ylim(),
        c='k',
    )
    lda_ax.set_ylabel('M1 LDA')
    lda_ax.set_xlabel(f'Time from {ref_event} (s)')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

def plot_hold_behavior(td_hold,label_col='task',hue_order=['CO','CST']):
    '''
    Plot out hold time behavior (hand position and velocity)

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        fig (Figure): Figure of behavior plot
    '''
    assert type(label_col)==str, "label_col must be a string"

    fig,[pos_ax,vel_ax] = plt.subplots(1,2,figsize=(6,4))
    sns.scatterplot(
        ax=pos_ax,
        data=td_hold,
        x=np.row_stack(td_hold['rel_hand_pos'].values)[:,0],
        y=np.row_stack(td_hold['rel_hand_pos'].values)[:,1],
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    pos_ax.set_aspect('equal')
    pos_ax.set_xlabel('X')
    pos_ax.set_ylabel('Y')
    pos_ax.set_title('Hand position (mm)')
    sns.scatterplot(
        ax=vel_ax,
        data=td_hold,
        x=np.row_stack(td_hold['hand_vel'].values)[:,0],
        y=np.row_stack(td_hold['hand_vel'].values)[:,1],
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    vel_ax.legend_.remove()
    vel_ax.set_aspect('equal')
    vel_ax.set_xlabel('X')
    vel_ax.set_title('Hand velocity (mm/s)')

    sns.despine(fig=fig,trim=True)

    return fig

def plot_M1_lda(td_hold,label_col='task',hue_order=['CO','CST']):
    '''
    Plot the M1 neural population activity LDA for CO and CST trials

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        lda_fig (Figure): Figure of lda plot

    TODO: add discriminability text somewhere in this plot
    '''
    # Hold-time LDA
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(6,4))
    lda_ax.plot([td_hold['trial_id'].min(),td_hold['trial_id'].max()],[0,0],'k--')
    sns.scatterplot(
        ax=lda_ax,
        data=td_hold,
        y='M1_lda',
        x='trial_id',
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    lda_ax.text(0,-3,'Discriminability: {:.2f}'.format((td_hold['M1_pred']==td_hold[label_col]).mean()))
    lda_ax.set_ylabel('M1 LDA projection')
    lda_ax.set_xlabel('Trial ID')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

def plot_beh_lda(td_hold,label_col='task',hue_order=['CO','CST']):
    '''
    Plot the behavioral LDA for CO and CST trials

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        lda_fig (Figure): Figure of lda plot
    '''
    # Hold-time LDA
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(6,4))
    lda_ax.plot([td_hold['trial_id'].min(),td_hold['trial_id'].max()],[0,0],'k--')
    sns.scatterplot(
        ax=lda_ax,
        data=td_hold,
        y='beh_lda',
        x='trial_id',
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    lda_ax.text(0,-3,'Discriminability: {:.2f}'.format((td_hold['beh_pred']==td_hold[label_col]).mean()))
    lda_ax.set_ylabel('Behavioral LDA projection')
    lda_ax.set_xlabel('Trial ID')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

def plot_M1_hold_potent(td,label_col='task',hue_order=['CO','CST']):
    '''
    Plot the M1 potent neural population activity for trials separated by label (e.g. task)

    Arguments:
        td (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        pca_fig (Figure): Figure of PCA plot
    '''
    assert type(label_col)==str, "label_col must be a string"

    pca_fig,pca_ax = plt.subplots(1,1,figsize=(6,4))
    sns.scatterplot(
        ax=pca_ax,
        data=td,
        x=np.row_stack(td['M1_potent_space'].values)[:,0],
        y=np.row_stack(td['M1_potent_space'].values)[:,1],
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    pca_ax.set_ylabel('M1 Potent 2')
    pca_ax.set_xlabel('M1 Potent 1')
    sns.despine(ax=pca_ax,trim=True)

    return pca_fig

def plot_M1_potent_lda(td_hold,label_col='task',hue_order=['CO','CST']):
    '''
    Plot the M1 neural population activity LDA (in potent space) for CO and CST trials

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        lda_fig (Figure): Figure of lda plot

    TODO: add discriminability text somewhere in this plot
    '''
    # Hold-time LDA
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(6,4))
    lda_ax.plot([td_hold['trial_id'].min(),td_hold['trial_id'].max()],[0,0],'k--')
    sns.scatterplot(
        ax=lda_ax,
        data=td_hold,
        y='M1_potent_lda',
        x='trial_id',
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    lda_ax.text(0,-3,'Discriminability: {:.2f}'.format((td_hold['M1_potent_pred']==td_hold[label_col]).mean()))
    lda_ax.set_ylabel('M1 potent space LDA projection')
    lda_ax.set_xlabel('Trial ID')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig

def plot_M1_null_lda(td_hold,label_col='task',hue_order=['CO','CST']):
    '''
    Plot the M1 neural population activity LDA (in null space) for CO and CST trials

    Arguments:
        td_hold (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        lda_fig (Figure): Figure of lda plot

    TODO: add discriminability text somewhere in this plot
    '''
    # Hold-time LDA
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(6,4))
    lda_ax.plot([td_hold['trial_id'].min(),td_hold['trial_id'].max()],[0,0],'k--')
    sns.scatterplot(
        ax=lda_ax,
        data=td_hold,
        y='M1_null_lda',
        x='trial_id',
        hue=label_col,palette='Set1',hue_order=hue_order
    )
    lda_ax.text(0,-3,'Discriminability: {:.2f}'.format((td_hold['M1_null_pred']==td_hold[label_col]).mean()))
    lda_ax.set_ylabel('M1 null space LDA projection')
    lda_ax.set_xlabel('Trial ID')
    sns.despine(ax=lda_ax,trim=True)

    return lda_fig