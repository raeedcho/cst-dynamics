#%%
import src

import pyaldata
import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from src.models import SSA

import seaborn as sns
import matplotlib.pyplot as plt
import k3d
import scipy.io as sio

with open("../params.yaml", "r") as params_file:
    lfads_params = yaml.safe_load(params_file)["lfads_prep"]

load_params = {
    'file_prefix': 'Prez_20220721',
    'verbose': False,
    'keep_unsorted': False,
    'lfads_params': lfads_params,
    'epoch_fun': src.util.generate_realtime_epoch_fun(
        start_point_name='idx_ctHoldTime',
        end_point_name='idx_endTime',
    ),
    'bin_size': 0.01,
}
joint_pca_model = src.models.JointSubspace(n_comps_per_cond=15,signal='lfads_rates',condition='task',remove_latent_offsets=False)
td = (
    src.data.load_clean_data(**load_params)
    .query('task=="RTT" | task=="CST"')
    .assign(**{'trialtime': lambda x: x['Time from go cue (s)']})
    .pipe(pyaldata.soft_normalize_signal,signals=['lfads_rates','MC_rates'])
    .pipe(src.data.remove_baseline_rates,signals=['MC_rates','lfads_rates'])
    .pipe(pyaldata.dim_reduce,model=TruncatedSVD(n_components=15),signal='MC_rates',out_fieldname='MC_svd')
    .pipe(pyaldata.dim_reduce,model=TruncatedSVD(n_components=15),signal='lfads_rates',out_fieldname='lfads_svd')
    .pipe(pyaldata.dim_reduce,model=PCA(n_components=15),signal='MC_rates',out_fieldname='MC_pca')
    .pipe(pyaldata.dim_reduce,model=PCA(n_components=15),signal='lfads_rates',out_fieldname='lfads_pca')
    .pipe(joint_pca_model.fit_transform)
)

# Data pipeline for decoding analysis
# - Soft normalize
# - Remove baseline

# trial_data = src.data.crystalize_dataframe(td,sig_guide={
#     'MC_rates': [f'ch{chan}u{unit}' for chan,unit in td['MC_unit_guide'].values[0]],
#     'lfads_rates': [f'ch{chan}u{unit}' for chan,unit in td['MC_unit_guide'].values[0]],
#     'lfads_inputs': None,
#     'rel_cursor_pos': None,
#     'rel_hand_pos': None,
#     'hand_vel': None,
#     'cursor_vel': None,
#     'hand_speed': None,
#     'hand_acc': None,
#     'Time from go cue (s)': None,
#     'Time from task cue (s)': None,
# })
# trial_info = src.data.extract_metaframe(td,metacols=['trial_id','task','lambda','ct_location','result','rt_locations'])
# full_td = trial_info.join(trial_data)

#%% Find joint subspace
# exp_td = src.data.explode_td(td)
# signal = 'lfads_rates'
# num_dims = 15
# joint_subspace_model = src.models.JointSubspace(n_comps_per_cond=num_dims).fit(np.row_stack(exp_td[signal]),exp_td['task'])
# td = td.assign(**{
#     signal.replace('rates','pca'): [joint_subspace_model.transform(s) for s in td[signal]]
# })

#%% Context space
signal = 'lfads_rates'
tonic_context_model = LinearDiscriminantAnalysis()
td_models = src.data.rebin_data(td,new_bin_size=0.100)
tonic_context_model.fit(
    np.row_stack(td_models.apply(lambda x: x[signal][x['idx_goCueTime']+15,:],axis=1)),
    td_models['task'],
)

transient_context_model = LinearDiscriminantAnalysis()
transient_context_model.fit(
    np.row_stack(td_models.apply(lambda x: x[signal][x['idx_pretaskHoldTime']+3,:],axis=1)),
    td_models['task'],
)

def norm_vec(vec):
    return vec/np.linalg.norm(vec)

td['Motor Cortex Tonic Context Dim'] = [(sig @ norm_vec(tonic_context_model.coef_).squeeze()[:,None]).squeeze() for sig in td[signal]]
td['Motor Cortex Transient Context Dim'] = [(sig @ norm_vec(transient_context_model.coef_).squeeze()[:,None]).squeeze() for sig in td[signal]]

td_explode = (
    td
    .assign(
        **{'Hand velocity (cm/s)': lambda x: x.apply(lambda y: y['hand_vel'][:,0],axis=1)}
    )
    .filter(items=[
        'trial_id',
        'Time from go cue (s)',
        'Time from task cue (s)',
        'task',
        'Motor Cortex Transient Context Dim',
        'Motor Cortex Tonic Context Dim',
        'Hand velocity (cm/s)'
    ])
    .explode([
        'Time from go cue (s)',
        'Time from task cue (s)',
        'Motor Cortex Transient Context Dim',
        'Motor Cortex Tonic Context Dim',
        'Hand velocity (cm/s)',
    ])
    .astype({
        'Time from go cue (s)': float,
        'Time from task cue (s)': float,
        'Motor Cortex Transient Context Dim': float,
        'Motor Cortex Tonic Context Dim': float,
        'Hand velocity (cm/s)': float,
    })
    # .loc[lambda df: df['Time from go cue (s)']>0]
    # .loc[lambda df: (df['Time from go cue (s)']<0) & (df['Time from go cue (s)']>-0.5)]
)
avg_trial = td_explode.groupby(['Time from go cue (s)','task']).mean().loc[-1:5].reset_index()
task_colors={'RTT': 'C1','CST': 'C0'}
fig,axs = plt.subplots(2,1,sharex=True,figsize=(6,6))
epoch = 'go'
for _,trial in td.groupby('task').sample(n=10).iterrows():
    # put an average trace over this thing
    axs[0].plot(
        trial[f'Time from {epoch} cue (s)'],
        trial['Motor Cortex Transient Context Dim'],
        color=task_colors[trial['task']],
        alpha=0.3,
        lw=2,
    )
    axs[1].plot(
        trial[f'Time from {epoch} cue (s)'],
        trial['Motor Cortex Tonic Context Dim'],
        color=task_colors[trial['task']],
        alpha=0.3,
        lw=2,
    )
    # axs.set_xlim([-1,5])
    # axs.set_ylim([-0.3,0.3])
    # axs.set_ylabel(f'Comp {compnum+1}')
for task,trial in avg_trial.groupby('task'):
    axs[0].plot(
        trial[f'Time from {epoch} cue (s)'],
        trial['Motor Cortex Transient Context Dim'],
        color=task_colors[task],
        lw=4,
    )
    axs[1].plot(
        trial[f'Time from {epoch} cue (s)'],
        trial['Motor Cortex Tonic Context Dim'],
        color=task_colors[task],
        lw=4,
    )
axs[0].set_ylabel('Motor Cortex\nContext Dim')
axs[1].set_ylabel('Motor Cortex\nContext Dim')
#axs[2].set_ylabel('Behavioral\nContext Dim')
axs[-1].set_xlabel(f'Time from {epoch} cue (s)')
axs[0].set_xlim([-1,5])
sns.despine(fig=fig,trim=True)

# %% decoding

trial_fig, score_fig = src.decoder_analysis.run_decoder_analysis(td, 'lfads_rates_joint_pca')

# %% plot individual traces

# %% Export data for subspace splitting in MATLAB
'''
This code will go through the following steps:
- Run a separated-rejoined PCA on CST and RTT data:
    - Run PCA separately on RTT and CST data in this epoch
    - Concatenate the PC weights from the two tasks
    - Run SVD on the concatenated PC weights to get new component weights
    - Project the data onto the new component weights
- Calculate the covariance matrices for each task
'''

signal = 'lfads_rates_joint_pca'
remove_task_mean_activity = True
num_dims = 15

if remove_task_mean_activity:
    covar_mats = (
        td
        .groupby('task')
        .apply(lambda df: pd.DataFrame(data=np.cov(np.row_stack(df[signal]),rowvar=False)))
    )
else:
    covar_mats = (
        td
        .groupby('task')
        .apply(lambda df: pd.DataFrame(data=np.row_stack(df[signal]).T @ np.row_stack(df[signal]) / df.shape[0]))
    )

for task in ['CST','RTT']:
    covar_mats.loc[task].to_csv(
        f"../results/subspace_splitting/Prez_20220721_{task}_{signal}_covar_mat.csv",
        header=False,
        index=False,
    )

# %% import subpsace splitter data

matfile = sio.loadmat(
    f"../results/subspace_splitting/Prez_20220721_CSTRTT_{signal}_subspacesplitter.mat",
    squeeze_me=True,
)

Q = {key: matfile['Q'][key].item() for key in matfile['Q'].dtype.names}
varexp = {key: matfile['varexp'][key].item() for key in matfile['varexp'].dtype.names}

# verify that varexp matches the covariance matrices we calculated
# np.trace(Q['unique1'].T @ covar_mats.loc['RTT'] @ Q['unique1'])/np.trace(covar_mats.loc['RTT'])

var_thresh = 0.016 # slightly arbitrary, chosen by looking at split variance explained numbers
cst_unique_proj = Q['unique1'][:,varexp['unique1_C1']>var_thresh]
rtt_unique_proj = Q['unique2'][:,varexp['unique2_C2']>var_thresh]
shared_proj = Q['shared']

# project data through the joint space into the split subspaces
td_proj = (
    td
    .assign(**{
        f'{signal}_cst_unique': lambda df: df.apply(lambda s: np.dot(s[signal],cst_unique_proj),axis=1),
        f'{signal}_rtt_unique': lambda df: df.apply(lambda s: np.dot(s[signal],rtt_unique_proj),axis=1),
        f'{signal}_shared': lambda df: df.apply(lambda s: np.dot(s[signal],shared_proj),axis=1),
    })
)

# %%
# A couple things to do:
#   - Check how aligned decoder axis is with each subspace
#   - Transfer these subspaces back to original data structure to plot hand position and target info

# %% Plot individual traces
def plot_trial_split_space(trial_to_plot,ax_list):
    src.plot.plot_hand_trace(trial_to_plot,ax=ax_list[0],timesig='Time from go cue (s)')
    src.plot.plot_hand_velocity(trial_to_plot,ax_list[1],timesig='Time from go cue (s)')

    sig_list = [f'{signal}_shared',f'{signal}_cst_unique',f'{signal}_rtt_unique']
    sig_colors = {
        f'{signal}_cst_unique':'C0',
        f'{signal}_rtt_unique':'C1',
        f'{signal}_shared': 'C4',
    }

    rownum = 2
    for sig in sig_list:
        for dim in range(trial[sig].shape[1]):
            ax = ax_list[rownum]
            ax.plot(trial_to_plot['Time from go cue (s)'][[0,-1]],[0,0],color='k')
            ax.plot(trial_to_plot['Time from go cue (s)'],trial_to_plot[sig][:,dim],color=sig_colors[sig])
            # ax.set_yticks([])
            ax.plot([0,0],ax.get_ylim(),color='k',linestyle='--')
            sns.despine(ax=ax,trim=True)
            rownum+=1

    ax_list[-1].set_xlabel('Time from go cue (s)')

trials_to_plot = td_proj.groupby('task').sample(n=1).set_index('trial_id')
fig,axs = plt.subplots(19,len(trials_to_plot),sharex=True,sharey='row',figsize=(10,18))
fig.tight_layout()
for colnum,(trial_id,trial) in enumerate(trials_to_plot.iterrows()):
    plot_trial_split_space(trial,axs[:,colnum])

# %% Plot average traces (they don't really make any sense though...)
td_subspace_split_avg = pyaldata.trial_average(td_subspace_split,condition='task',ref_field='lfads_pca')
fig,axs = plt.subplots(19,len(td_subspace_split_avg),sharex=True,sharey='row',figsize=(10,18))
fig.tight_layout()
for colnum,(task,trial) in enumerate(td_subspace_split_avg.iterrows()):
    plot_trial_split_space(trial,axs[:,colnum])

#%% k3d plots
cst_trace_plot = k3d.plot(name='CST smoothed neural traces')
max_abs_hand_vel = np.percentile(np.abs(np.row_stack(td['hand_vel'])[:,0]),95)
# plot traces
for _,trial in td.query('task=="CST"').sample(n=10).iterrows():
    neural_trace = trial['lfads_pca']
    cst_trace_plot+=k3d.line(
        neural_trace[:,0:3].astype(np.float32),
        shader='mesh',
        width=3e-3,
        attribute=trial['hand_vel'][:,0],
        color_map=k3d.paraview_color_maps.Erdc_divHi_purpleGreen,
        color_range=[-max_abs_hand_vel,max_abs_hand_vel],
    )
cst_trace_plot.display()

rtt_trace_plot = k3d.plot(name='RTT smoothed neural traces')
for _,trial in td.query('task=="RTT"').sample(n=10).iterrows():
    neural_trace = trial['lfads_pca']
    rtt_trace_plot+=k3d.line(
        neural_trace[:,0:3].astype(np.float32),
        shader='mesh',
        width=3e-3,
        attribute=trial['hand_vel'][:,0],
        color_map=k3d.paraview_color_maps.Erdc_divHi_purpleGreen,
        color_range=[-max_abs_hand_vel,max_abs_hand_vel],
    )
rtt_trace_plot.display()

#%% k3d plots with explode-y td
max_abs_hand_vel = np.percentile(np.abs(td_proj['Hand velocity (cm/s)']),95)

def plot_k3d_trace(trial,plot,color=None):
    neural_trace = np.row_stack(trial[f'{signal}_shared'])
    plot+=k3d.line(
        neural_trace[:,0:3].astype(np.float32),
        shader='mesh',
        width=3e-3,
        color=color,
        # attribute=trial['Hand velocity (cm/s)'],
        # color_map=k3d.paraview_color_maps.Erdc_divHi_purpleGreen,
        # color_range=[-max_abs_hand_vel,max_abs_hand_vel],
    )
    plot.display()

# plot traces
cst_trial = td_proj.loc[227]
rtt_trial = td_proj.loc[228]
# cst_trace_plot = k3d.plot(name='CST neural traces in shared space')
# rtt_trace_plot = k3d.plot(name='RTT neural traces in shared space')
trials_plot = k3d.plot(name='CST and RTT trials in CST subspace')
plot_k3d_trace(cst_trial,trials_plot,color=0x1f77b4)
plot_k3d_trace(rtt_trial,trials_plot,color=0xff7f0e)
