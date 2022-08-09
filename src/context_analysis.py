import pyaldata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import subspace_tools,data,util

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

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
        .pipe(pyaldata.add_firing_rates,method='bin')
    )

    spike_fields = [name for name in td.columns.values if name.endswith("_spikes")]
    for field in spike_fields:
        assert td_binned[field].values[0].ndim==1, "Binning didn't work"

    smooth_epoch_dict = {
        'hold_move': src.util.generate_realtime_epoch_fun(
            'idx_goCueTime',
            rel_start_time=-0.8,
            rel_end_time=0.5,
        ),
        'hold_move_ref_cue': src.util.generate_realtime_epoch_fun(
            'idx_pretaskHoldTime',
            rel_start_time=-0.3,
            rel_end_time=1.0,
        ),
        'full': lambda trial : slice(0,trial['hand_pos'].shape[0]),
    }
    td_smooth = (
        td.copy()
        .pipe(pyaldata.add_firing_rates,method='smooth',std=0.05,backend='convolve')
        .pipe(src.util.split_trials_by_epoch,smooth_epoch_dict)
        .pipe(src.data.rebin_data,new_bin_size=0.05)
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

    rate_fields = [name for name in td_train.columns if name.endswith('_rates')]
    for field in rate_fields:
        pca_model = PCA(n_components=15)
        td_train[field.replace('rates','pca')] = list(pca_model.fit_transform(np.row_stack(td_train[field])))
        # td_train[field.replace('rates','pca')] = [pca_model.transform(rates) for rates in td_train[field]]
        td_test[field.replace('rates','pca')] = [pca_model.transform(rates) for rates in td_test[field]]

        M1_lda_model = LinearDiscriminantAnalysis()
        td_train[field.replace('rates','lda')] = M1_lda_model.fit_transform(
            np.row_stack(td_train[field.replace('rates','pca')].values),
            td_train[label_col]
        )
        td_train[field.replace('rates','pred')] = M1_lda_model.predict(np.row_stack(td_train[field.replace('rates','pca')]))
        td_test[field.replace('rates','lda')] = [M1_lda_model.transform(sig) for sig in td_test[field.replace('rates','pca')]]

        # check separability in neuron-behavioral potent space
        potent_space,null_space = subspace_tools.find_potent_null_space(
            np.row_stack(td_train[field.replace('rates','pca')]),
            np.column_stack([
                np.row_stack(td_train['rel_hand_pos'].values),
                np.row_stack(td_train['hand_vel'].values),
            ]),
        )
        td_train[field.replace('rates','potent_space')] = [neural_state @ potent_space for neural_state in td_train[field.replace('rates','pca')]]
        td_test[field.replace('rates','potent_space')] = [neural_state @ potent_space for neural_state in td_test[field.replace('rates','pca')]]
        M1_potent_lda_model = LinearDiscriminantAnalysis()
        td_train[field.replace('rates','potent_lda')] = M1_potent_lda_model.fit_transform(
            np.row_stack(td_train[field.replace('rates','potent_space')].values),
            td_train[label_col]
        )
        td_train[field.replace('rates','potent_pred')] = M1_potent_lda_model.predict(np.row_stack(td_train[field.replace('rates','potent_space')]))
        td_test[field.replace('rates','potent_lda')] = [M1_potent_lda_model.transform(sig) for sig in td_test[field.replace('rates','potent_space')]]

        td_train[field.replace('rates','null_space')] = [neural_state @ null_space for neural_state in td_train[field.replace('rates','pca')]]
        td_test[field.replace('rates','null_space')] = [neural_state @ null_space for neural_state in td_test[field.replace('rates','pca')]]
        M1_null_lda_model = LinearDiscriminantAnalysis()
        td_train[field.replace('rates','null_lda')] = M1_null_lda_model.fit_transform(
            np.row_stack(td_train[field.replace('rates','null_space')].values),
            td_train[label_col]
        )
        td_train[field.replace('rates','null_pred')] = M1_null_lda_model.predict(np.row_stack(td_train[field.replace('rates','null_space')]))
        td_test[field.replace('rates','null_lda')] = [M1_null_lda_model.transform(sig) for sig in td_test[field.replace('rates','null_space')]]

    return td_train,td_test

def plot_hold_pca(td,array_name='M1',label_col='task',hue_order=['CO','CST']):
    '''
    Plot the M1 neural population activity for trials separated by label (e.g. task)

    Arguments:
        td (DataFrame): PyalData formatted structure of neural/behavioral data

    Returns:
        pca_fig (Figure): Figure of PCA plot
    '''
    assert type(label_col)==str, "label_col must be a string"

    pca_fig,pca_ax = plt.subplots(1,1,figsize=(8,6))
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

    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
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

    fig,[pos_ax,vel_ax] = plt.subplots(1,2,figsize=(10,6))
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
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
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
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
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

    pca_fig,pca_ax = plt.subplots(1,1,figsize=(8,6))
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
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
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
    lda_fig,lda_ax = plt.subplots(1,1,figsize=(8,6))
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