'''
Script to generate figures for the 2022 SfN poster. The structure of the poster will be:

- Big question: How do we generate movements that are continuously modified by feedback?
  - moreover, how do we study these data when every such trial is different and we can't use averages?
- We compared two tasks: CST and RTT (task explanation and some trial traces)
- Even though the goal of CST is simple, movements are nuanced (see Mohsen's poster for some extra analysis of this)
    (maybe change this one out for a comparison of behavior between CST and RTT, including histograms)
- We recorded neural activity from M1 and PMd (show rasters, mean FR comparison, Left/right selectivity comparison)
- Neural manifolds are similar across tasks (Similar dimensionality, high subspace overlap, shared behavioral subspace)
- Neural activity occupies different regions of space depending on the task demands

So the figures I need to generate:
- sensorimotor plots for delayed LTI control
- sensorimotor plot for example CST trial
x Example trial traces for CST and RTT
x Histograms of hand position and velocity for CST and RTT
x Example rasters for CST and RTT
x Mean FR scatterplot between CST and RTT for all neurons
x L/R selectivity scatterplot between CST and RTT for all neurons
x Scree plots for CST and RTT
x Subspace overlap between CST and RTT movement period
~ Behavioral subspace overlap between CST and RTT
- Context subspace dimensions
- Separability traces along each dimension? (classification accuracy of simple threshold?)
'''

#%% Setup
from tracemalloc import start
import src
import pyaldata
import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ipywidgets import interact
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle,Circle
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
sns.set_context('talk')

#! %load_ext autoreload
#! %autoreload 2

params = {
    'verbose': True,
    'keep_unsorted': True,
    'bin_size': 0.010,
}

filename = '../data/trial_data/Prez_20220720_RTTCSTCO_TD.mat'
td = (
    pyaldata.mat2dataframe(
        filename,
        shift_idx_fields=True,
        td_name='trial_data'
    )
    .assign(
        date_time=lambda x: pd.to_datetime(x['date_time']),
        session_date=lambda x: pd.DatetimeIndex(x['date_time']).normalize()
    )
    .query('task=="RTT" | task=="CST"')
    .pipe(src.data.remove_aborts, verbose=params['verbose'])
    .pipe(src.data.remove_artifact_trials, verbose=params['verbose'])
    .pipe(
        src.data.filter_unit_guides,
        filter_func=lambda guide: guide[:,1] > (0 if params['keep_unsorted'] else 1)
    )
    .pipe(src.data.remove_correlated_units)
    .pipe(
        src.data.remove_all_low_firing_neurons,
        threshold=0.1,
        divide_by_bin_size=True,
        verbose=params['verbose']
    )
    .pipe(pyaldata.add_firing_rates,method='smooth', std=0.05, backend='convolve')
    .pipe(src.data.trim_nans, ref_signals=['rel_hand_pos'])
    .pipe(src.data.fill_kinematic_signals)
    .pipe(src.data.rebin_data,new_bin_size=params['bin_size'])
    .pipe(pyaldata.soft_normalize_signal,signals=['M1_rates','PMd_rates','MC_rates'])
    .pipe(pyaldata.dim_reduce,PCA(n_components=15),'M1_rates','M1_pca')
    .pipe(pyaldata.dim_reduce,PCA(n_components=15),'PMd_rates','PMd_pca')
    .pipe(pyaldata.dim_reduce,PCA(n_components=15),'MC_rates','MC_pca')
    .assign(idx_ctHoldTime= lambda x: x['idx_ctHoldTime'].map(lambda y: y[-1] if y.size>1 else y))
    .astype({
        'idx_ctHoldTime': int,
        'idx_pretaskHoldTime': int,
        'idx_goCueTime': int,
    })
    .pipe(
        pyaldata.restrict_to_interval,
        epoch_fun = src.util.generate_realtime_epoch_fun(
            start_point_name='idx_ctHoldTime',
            end_point_name='idx_endTime',
        ),
        warn_per_trial=True,
    )
    .pipe(src.data.add_trial_time,ref_event='idx_goCueTime',column_name='Time from go cue (s)')
    .pipe(src.data.add_trial_time,ref_event='idx_pretaskHoldTime',column_name='Time from task cue (s)')
)
#     .pipe(
#         pyaldata.restrict_to_interval,
#         epoch_fun = src.util.generate_realtime_epoch_fun(
#             start_point_name='idx_goCueTime',
#             rel_start_time=-0.8,
#             end_point_name='idx_endTime',
#         ),
#         warn_per_trial=True,
#     )

#%% Behavioral traces
vel_model = LinearRegression(fit_intercept=False)
# context_model = LinearDiscriminantAnalysis()

td_models = src.data.rebin_data(td,new_bin_size=0.100)

vel_model.fit(
    np.row_stack(td_models['MC_pca']),
    np.row_stack(td_models['hand_vel'])[:,0],
)

# context_model.fit(
#     np.row_stack(td_models.apply(lambda x: x['MC_pca'][20,:],axis=1)),
#     td_models['task'],
# )

def norm_vec(vec):
    return vec/np.linalg.norm(vec)

td['Motor Cortex Velocity Dim'] = [(sig @ norm_vec(vel_model.coef_).squeeze()[:,None]).squeeze() for sig in td['MC_pca']]
# td['Motor Cortex Context Dim'] = [(sig @ norm_vec(context_model.coef_).squeeze()[:,None]).squeeze() for sig in td['MC_pca']]

# print(f'Angle between velocity dim and context dim: {src.util.angle_between(vel_model.coef_.squeeze(),context_model.coef_.squeeze())} degrees')

def plot_trial(trial):
    fig,axs=plt.subplots(4,1,sharex='col',sharey='row',figsize=(6,8))

    go_cue_trial = trial.copy()
    go_cue_trial['trialtime'] = trial['Time from go cue (s)']
    
    src.plot.plot_hand_trace(go_cue_trial,ax=axs[0])
    axs[0].set_ylabel('Hand position (cm)')
    src.plot.make_trial_raster(trial,ax=axs[1],sig='MC_spikes',ref_event_idx=trial['idx_goCueTime'])
    axs[1].set_ylabel('Neurons')

    # hand velocity
    src.plot.plot_hand_velocity(go_cue_trial,axs[2])
    axs[2].set_ylim([-300,300])
    axs[2].set_ylabel('Hand velocity (cm/s)')

    # neural dimension
    axs[3].plot(
        go_cue_trial['trialtime'],
        go_cue_trial['Motor Cortex Velocity Dim'],
        color=[0.5,0.5,0.5],
    )
    axs[3].set_yticks([])
    axs[3].set_ylabel('Motor cortex\nvelocity dim')

    axs[3].set_xlim([-1,6])
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')
    axs[3].set_xlabel('Time from go cue (s)')
    fig.suptitle(f'Trial {trial["trial_id"]} ({trial["task"]})')

    sns.despine(fig=fig,trim=True)
    sns.despine(ax=axs[1],trim=True,left=True)
    sns.despine(ax=axs[3],trim=True,left=True)
    return fig

# plot_trial(td.loc[td['trial_id']==228].squeeze())
# plt.tight_layout()
fig_name = src.util.format_outfile_name(td,postfix='trial_traces')
with PdfPages(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf')) as pdf:
    for _,trial in td.iterrows():
        fig = plot_trial(trial)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

#%% Position and velocity histograms for CST and RTT
td_explode = (
    td
    .assign(**{
        'Hand position (cm)': lambda x: x.apply(lambda y: y['rel_hand_pos'][:,0],axis=1),
        'Hand velocity (cm/s)': lambda x: x.apply(lambda y: y['hand_vel'][:,0],axis=1),
    })
    .loc[:,['trial_id','Time from go cue (s)','task','Motor Cortex Velocity Dim','Hand velocity (cm/s)','Hand position (cm)']]
    .explode(['Time from go cue (s)','Hand velocity (cm/s)','Hand position (cm)','Motor Cortex Velocity Dim'])
    .astype({
        'Time from go cue (s)': float,
        'Motor Cortex Velocity Dim': float,
        'Hand position (cm)': float,
        'Hand velocity (cm/s)': float,
    })
    .set_index(['trial_id','Time from go cue (s)'])
)

fig,axs = plt.subplots(2,1,figsize=(6,8))
sns.kdeplot(
    data=td_explode,
    y='Hand position (cm)',
    hue='task',
    ax=axs[0]
)
axs[0].set_ylim([-60,60])
axs[0].set_xticks([])
axs[0].set_xlabel('')
axs[0].get_legend().remove()

sns.kdeplot(
    data=td_explode,
    y='Hand velocity (cm/s)',
    hue='task',
    ax=axs[1]
)
axs[1].set_ylim([-300,300])
axs[1].set_xticks([])

sns.despine(fig=fig,trim=True,bottom=True)
fig_name = src.util.format_outfile_name(td,postfix='cst_rtt_beh_histograms')
fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

#%% Single neuron analyses
td_mean_fr = (
    td
    .pipe(src.data.rebin_data,new_bin_size=0.05)
    .assign(**{
        'Hand position (cm)': lambda x: x.apply(lambda y: y['rel_hand_pos'][:,0],axis=1),
        'Hand velocity (cm/s)': lambda x: x.apply(lambda y: y['hand_vel'][:,0],axis=1),
    })
    .filter(items=[
        'trial_id',
        'Time from go cue (s)',
        'task',
        'Hand velocity (cm/s)',
        'Hand position (cm)',
        'MC_unit_guide',
        'MC_spikes'
    ])
    .explode(['Time from go cue (s)','Hand velocity (cm/s)','Hand position (cm)','MC_spikes'])
    .astype({
        'Time from go cue (s)': float,
        'Hand position (cm)': float,
        'Hand velocity (cm/s)': float,
    })
    .loc[lambda df: df['Time from go cue (s)']>=0]
    .groupby('task')
    .apply( lambda x: pd.DataFrame({
        'mean_fr': np.mean(x['MC_spikes'])/0.05,
        'lr_selectivity': src.single_neuron_analysis.get_lr_selectivity(
            np.row_stack(x['Hand velocity (cm/s)']),
            np.row_stack(x['MC_spikes'])/0.05,
        )
    }))
)

fig,ax = plt.subplots(1,1,figsize=(6,6))
ax.plot([0,100],[0,100],'--k')
ax.scatter(
    td_mean_fr.loc['CST','mean_fr'],
    td_mean_fr.loc['RTT','mean_fr'],
    color='k',
)
ax.set_xlabel('CST mean firing rate (Hz)')
ax.set_ylabel('RTT mean firing rate (Hz)')
sns.despine(ax=ax,trim=True)
fig_name = src.util.format_outfile_name(td,postfix='cst_rtt_mean_fr_comparison')
fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

fig,ax = plt.subplots(1,1,figsize=(6,6))
ax.plot([0,0],[-0.005,0.005],'-k')
ax.plot([-0.005,0.005],[0,0],'-k')
ax.plot([-0.005,0.005],[-0.005,0.005],'--k')
ax.scatter(
    td_mean_fr.loc['CST','lr_selectivity'],
    td_mean_fr.loc['RTT','lr_selectivity'],
    color='k',
)
ax.set_xlabel('CST rightward tuning (Hz/(cm/s))')
ax.set_ylabel('RTT rightward tuning (Hz/(cm/s))')
sns.despine(ax=ax,trim=True)
fig_name = src.util.format_outfile_name(td,postfix='cst_rtt_lr_selectivity_comparison')
fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

#%% PCA scree plot
def get_scree(arr):
    '''
    Get the fraction of variance explained by each component of PCA
    
    Args:
        arr (np.array): array to perform PCA on, shape (n_timepoints,n_features)
        
    Returns:
        (np.array): array of fraction variance explained by each component, shape (n_features,)
    '''
    model = PCA()
    model.fit(arr)
    return model.explained_variance_ratio_

td_scree = (
    td
    .filter(items=['trial_id','task','Time from go cue (s)','MC_rates'])
    .explode(['Time from go cue (s)','MC_rates'])
    .loc[lambda df: df['Time from go cue (s)']>0]
    .groupby('task')
    .agg(
        compnum=('MC_rates',lambda x: 1+np.arange(x.values[0].shape[0])),
        scree=('MC_rates',lambda x: get_scree(np.row_stack(x))),
    )
    .explode(['compnum','scree'])
    .reset_index()
)
participation_ratio = (
    td_scree
    .groupby('task')
    .apply(
        lambda x: (np.sum(x['scree'])**2)/np.sum(x['scree']**2)
    )
)
print(participation_ratio)

fig,ax = plt.subplots(1,1)
sns.lineplot(
    ax=ax,
    data=td_scree,
    x='compnum',
    y='scree',
    hue='task',
    hue_order=['CST','RTT'],
)
ax.set_ylabel('Fraction variance explained')
ax.set_xlabel('Component #')
# ax.set(xscale='log',yscale='log')
# sub_ax = fig.add_axes((0.4,0.5,0.25,0.25))
# sns.lineplot(
#     ax=sub_ax,
#     data=td_scree,
#     x='compnum',
#     y='scree',
#     hue='task',
#     hue_order=['CST','RTT'],
#     legend=False,
# )
# sub_ax.set_xlim([0,25])
# sub_ax.set_ylabel('')
# sub_ax.set_xlabel('')
sns.despine(fig=fig,trim=True)
fig_name = src.util.format_outfile_name(td,postfix='cst_rtt_scree_plot')
fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

#%% subspace overlap
td_subspace_overlap = (
    td
    .pipe(pyaldata.restrict_to_interval,epoch_fun=src.util.generate_realtime_epoch_fun(
        start_point_name='idx_goCueTime',
        end_point_name='idx_endTime',
    ))
    .pipe(src.data.rebin_data,new_bin_size=0.05)
    .groupby('task',as_index=False)
    .pipe(src.subspace_tools.bootstrap_subspace_overlap,signal='MC_rates',num_bootstraps=100)
    .set_index(['task_data','task_proj','boot_id'])
    .sort_index()
)

fig,ax = plt.subplots(1,1)
sns.histplot(
    ax=ax,
    data=td_subspace_overlap.loc[('CST','RTT')],
    x='subspace_overlap',
    color='k',
    edgecolor=None,
)
sns.histplot(
    ax=ax,
    data=td_subspace_overlap.loc[('CST','RTT')],
    x='subspace_overlap_rand',
    color='0.7',
    edgecolor=None,
)
# ax.text(15,0.3,'CST->RTT',color='k')
ax.text(0.25,10,'Random\nmanifolds',color='0.7',fontsize=18)
ax.set_xlim([0,1])
ax.set_xlabel('Subspace overlap CST->RTT')
sns.despine(ax=ax,trim=True)
fig_name = src.util.format_outfile_name(td,postfix='cst_rtt_subspace_overlap')
fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

#%% Behavioral subspace similarity (how similar are coefficients of behavioral projection?)
def get_beh_coefs(df):
    '''
    Run a linear regression from neural activity to behavior to get coefficients

    Args:
        df (pd.DataFrame): pyaldata format dataframe containing neural activity and behavior
    Returns:
        (np.array): array of coefficients
    '''
    beh_model = LinearRegression()

    beh_model.fit(
        np.row_stack(df['MC_pca']),
        np.column_stack([
            np.row_stack(df['hand_pos']),
            np.row_stack(df['hand_vel']),
        ]),
    )
    return beh_model.coef_

beh_coefs = (
    td
    .pipe(pyaldata.restrict_to_interval,epoch_fun=src.util.generate_realtime_epoch_fun(
        start_point_name='idx_goCueTime',
        end_point_name='idx_endTime',
    ))
    .pipe(src.data.rebin_data,new_bin_size=0.100)
    .groupby('task')
    .apply(get_beh_coefs)
)

#%% Neural dimensions
td_explode = (
    td
    .assign(
        **{'Hand velocity (cm/s)': lambda x: x.apply(lambda y: y['hand_vel'][:,0],axis=1)}
    )
    .loc[:,['trial_id','Time from go cue (s)','Time from task cue (s)','task','Motor Cortex Velocity Dim','Motor Cortex Context Dim','Hand velocity (cm/s)']]
    .explode(['trialtime','Hand velocity (cm/s)','Motor Cortex Velocity Dim','Motor Cortex Context Dim'])
    .astype({
        'trialtime': float,
        'Motor Cortex Velocity Dim': float,
        'Motor Cortex Context Dim': float,
        'Hand velocity (cm/s)': float,
    })
)
g = sns.pairplot(
    data=td_explode.sample(n=250),
    x_vars='Hand velocity (cm/s)',
    y_vars=['Motor Cortex Velocity Dim','Motor Cortex Context Dim'],
    hue='task',
    hue_order=['CST','RTT'],
    kind='reg',
    height=4,
    aspect=1,
)
sns.despine(fig=g.fig,trim=True)
fig_name = src.util.format_outfile_name(td,postfix='neural_dims_v_vel')
g.fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

#%% Context space
avg_trial = td_explode.groupby(['trialtime','task']).mean().loc[:5].reset_index()
task_colors={'RTT': 'tab:orange','CST': 'tab:blue'}
fig,axs = plt.subplots(2,1,sharex=True,figsize=(6,6))
for _,trial in td.groupby('task').sample(n=25).iterrows():
    axs[0].plot(
        trial['trialtime'],
        trial['Motor Cortex Velocity Dim'],
        color=task_colors[trial['task']],
        alpha=0.3,
        lw=2,
    )
    # put an average trace over this thing
    axs[1].plot(
        trial['trialtime'],
        trial['Motor Cortex Context Dim'],
        color=task_colors[trial['task']],
        alpha=0.3,
        lw=2,
    )
    # axs.set_xlim([-1,5])
    # axs.set_ylim([-0.3,0.3])
    # axs.set_ylabel(f'Comp {compnum+1}')
for task,trial in avg_trial.groupby('task'):
    axs[0].plot(
        trial['trialtime'],
        trial['Motor Cortex Velocity Dim'],
        color=task_colors[task],
        lw=4,
    )
    axs[1].plot(
        trial['trialtime'],
        trial['Motor Cortex Context Dim'],
        color=task_colors[task],
        lw=4,
    )
axs[0].set_ylabel('Motor Cortex\nVelocity Dim')
axs[1].set_ylabel('Motor Cortex\nContext Dim')
axs[-1].set_xlabel('Time from go cue (s)')
sns.despine(fig=fig,trim=True)
fig_name = src.util.format_outfile_name(td,postfix='neural_dims_v_time')
fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

# %% Trial animation (hand movement, raster, PC1 v PC2 population)
def animate_trial_timecourse(trial):
    fig = plt.figure(figsize=(10,6))
    gs = mpl.gridspec.GridSpec(2,2,figure=fig)
    beh_ax = fig.add_subplot(gs[0,0])
    raster_ax = fig.add_subplot(gs[1,0])
    pop_ax = fig.add_subplot(gs[:,1])

    src.plot.plot_hand_trace(trial,ax=beh_ax)
    beh_blocker = beh_ax.add_patch(Rectangle((0,-100),10,200,color='w',zorder=100))
    beh_ax.set_xlim([0,6])
    beh_ax.set_xticks([])
    beh_ax.set_xlabel('')
    beh_ax.set_ylabel('Hand position (cm)')
    sns.despine(ax=beh_ax,trim=True,bottom=True)

    src.plot.make_trial_raster(trial,ax=raster_ax,sig='MC_spikes')
    raster_blocker = raster_ax.add_patch(Rectangle((0,-100),10,200,color='w',zorder=100))
    raster_ax.set_ylim([0,88])
    raster_ax.set_xlim([0,6])
    raster_ax.set_ylabel('Neurons')
    raster_ax.set_xlabel('Time from go cue (s)')

    pop_ax.plot([-0.4,-0.15],[-0.4,-0.4],color='k',lw=5)
    pop_ax.text(-0.35,-0.45,'PC1',fontsize=18)
    pop_ax.plot([-0.4,-0.4],[-0.4,-0.15],color='k',lw=5)
    pop_ax.text(-0.475,-0.3,'PC2',fontsize=18,rotation=90)
    pop_trace, = pop_ax.plot([],[],color='k')
    pop_ax.set_xlim([-0.5,0.5])
    pop_ax.set_ylim([-0.5,0.5])
    pop_ax.set_xticks([])
    pop_ax.set_yticks([])
    sns.despine(ax=pop_ax,left=True,bottom=True)

    plt.tight_layout()

    def plot_trial_timecourse(trial,end_idx=None):
        beh_blocker.set(x=trial['trialtime'][end_idx])
        raster_blocker.set(x=trial['trialtime'][end_idx])
    
        # first two PCs
        pop_trace.set_data(
            trial['MC_pca'][:end_idx,0],
            trial['MC_pca'][:end_idx,1],
        )
        return [beh_blocker,raster_blocker,pop_trace]

    def init_plot():
        beh_blocker.set(x=0)
        raster_blocker.set(x=0)
        pop_trace.set_data([],[])
        return [beh_blocker,raster_blocker,pop_trace]

    def animate(frame_time):
        epoch_fun = src.util.generate_realtime_epoch_fun(
            start_point_name='idx_goCueTime',
            rel_end_time=frame_time,
        )
        anim_slice = epoch_fun(trial)

        return plot_trial_timecourse(trial,end_idx=anim_slice.stop)

    frame_interval = 30 #ms
    frames = np.arange(trial['trialtime'][0],trial['trialtime'][-1],frame_interval*1e-3)
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init_plot,
        frames = frames,
        interval = frame_interval,
        blit = True,
    )

    return anim

trial_to_plot=228
anim = animate_trial_timecourse(td.loc[td['trial_id']==trial_to_plot].squeeze())
anim_name = src.util.format_outfile_name(td,postfix=f'trial_{trial_to_plot}_anim')
anim.save(os.path.join('../results/2022_sfn_poster/',anim_name+'.mp4'),writer='ffmpeg',fps=30,dpi=400)

# %% Animate comparison of CST vs RTT example trials
def animate_context_vel_dims(cst_trial,rtt_trial):
    fig = plt.figure(figsize=(10,6))
    gs = mpl.gridspec.GridSpec(2,2,figure=fig)
    rtt_ax = fig.add_subplot(gs[0,0])
    cst_ax = fig.add_subplot(gs[1,0])
    pop_ax = fig.add_subplot(gs[:,1])

    src.plot.plot_hand_trace(rtt_trial,ax=rtt_ax)
    rtt_blocker = rtt_ax.add_patch(Rectangle((0,-100),10,200,color='w',zorder=100))
    rtt_ax.set_xlim([0,6])
    rtt_ax.set_xticks([])
    rtt_ax.set_xlabel('')
    rtt_ax.set_ylabel('RTT\nHand position (cm)')
    sns.despine(ax=rtt_ax,trim=True,bottom=True)

    src.plot.plot_hand_trace(cst_trial,ax=cst_ax)
    cst_blocker = cst_ax.add_patch(Rectangle((0,-100),10,200,color='w',zorder=100))
    cst_ax.set_xlim([0,6])
    cst_ax.set_ylabel('CST\nHand position (cm)')
    sns.despine(ax=cst_ax,trim=True)

    pop_ax.plot([-0.4,-0.15],[-0.4,-0.4],color='k',lw=5)
    pop_ax.text(-0.4,-0.45,'Velocity dim',fontsize=18)
    pop_ax.plot([-0.4,-0.4],[-0.4,-0.15],color='k',lw=5)
    pop_ax.text(-0.475,-0.4,'Context dim',fontsize=18,rotation=90)
    pop_ax.text(0,-0.3,'CST',fontsize=21,color='C0')
    pop_ax.text(0,0.3,'RTT',fontsize=21,color='C1')
    cst_pop_trace, = pop_ax.plot([],[],color='C0')
    rtt_pop_trace, = pop_ax.plot([],[],color='C1')
    pop_ax.set_xlim([-0.5,0.5])
    pop_ax.set_ylim([-0.5,0.5])
    pop_ax.set_xticks([])
    pop_ax.set_yticks([])
    sns.despine(ax=pop_ax,left=True,bottom=True)
    plt.tight_layout()

    def init_plot():
        rtt_blocker.set(x=0)
        cst_blocker.set(x=0)
        cst_pop_trace.set_data([],[])
        rtt_pop_trace.set_data([],[])

        return [rtt_blocker,cst_blocker,cst_pop_trace,rtt_pop_trace]

    def animate(frame_time):
        epoch_fun = src.util.generate_realtime_epoch_fun(
            start_point_name='idx_goCueTime',
            rel_end_time=frame_time,
        )
        cst_anim_slice = epoch_fun(cst_trial)
        rtt_anim_slice = epoch_fun(rtt_trial)

        rtt_blocker.set(x=frame_time)
        cst_blocker.set(x=frame_time)
        cst_pop_trace.set_data(
            cst_trial['Motor Cortex Velocity Dim'][cst_anim_slice],
            cst_trial['Motor Cortex Context Dim'][cst_anim_slice],
        )
        rtt_pop_trace.set_data(
            rtt_trial['Motor Cortex Velocity Dim'][rtt_anim_slice],
            rtt_trial['Motor Cortex Context Dim'][rtt_anim_slice],
        )

        return [rtt_blocker,cst_blocker,cst_pop_trace,rtt_pop_trace]

    frame_interval = 30 #ms
    frames = np.arange(0,7,frame_interval*1e-3)
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init_plot,
        frames = frames,
        interval = frame_interval,
        blit = True,
    )
    return anim
    
rtt_trial_id = 227
cst_trial_id = 228
anim = animate_context_vel_dims(
    cst_trial = td.loc[td['trial_id']==cst_trial_id].squeeze(),
    rtt_trial = td.loc[td['trial_id']==rtt_trial_id].squeeze(),
)
anim_name = src.util.format_outfile_name(td,postfix=f'cst_trial_{cst_trial_id}_rtt_trial_{rtt_trial_id}_context_vel_dims_anim')
anim.save(os.path.join('../results/2022_sfn_poster/',anim_name+'.mp4'),writer='ffmpeg',fps=30,dpi=400)

# %% Animation for a single trial's behavior
def animate_trial_monitor(trial):
    fig = plt.figure(figsize=(10,6))
    gs = mpl.gridspec.GridSpec(2,2,figure=fig)
    monitor_ax = fig.add_subplot(gs[:,0])
    beh_ax = fig.add_subplot(gs[0,1])
    raster_ax = fig.add_subplot(gs[1,1])

    src.plot.plot_hand_trace(trial,ax=beh_ax)
    beh_blocker = beh_ax.add_patch(Rectangle((0,-100),10,200,color='w',zorder=100))
    beh_ax.set_xlim([0,6])
    beh_ax.set_xticks([])
    beh_ax.set_xlabel('')
    beh_ax.set_ylabel('Hand position (cm)')
    sns.despine(ax=beh_ax,trim=True,bottom=True)

    src.plot.make_trial_raster(trial,ax=raster_ax,sig='MC_spikes')
    raster_blocker = raster_ax.add_patch(Rectangle((0,-100),10,200,color='w',zorder=100))
    raster_ax.set_ylim([0,88])
    raster_ax.set_xlim([0,6])
    raster_ax.set_ylabel('Neurons')
    raster_ax.set_xlabel('Time from go cue (s)')

    hand = monitor_ax.add_patch(Circle(trial['rel_hand_pos'][0,:2],5,color='r',fill=False))
    cursor = monitor_ax.add_patch(Circle(trial['rel_cursor_pos'][0,:2], 5,zorder=100))
    if trial['task'] == 'RTT':
        cursor.set(color='y')
        targets = [
            monitor_ax.add_patch(Rectangle(
                targ_loc[:2]-trial['ct_location'][:2]-[5,5],
                10,
                10,
                color='r',
                visible=False,
            )) for targ_loc in trial['rt_locations']
        ]
    else:
        cursor.set(color='b')
        targets=[
            monitor_ax.add_patch(Rectangle((-5,-5),10,10,color='0.25'))
        ]

    monitor_ax.set_xlim([-60,60])
    monitor_ax.set_ylim([-60,60])
    monitor_ax.set_xticks([])
    monitor_ax.set_yticks([])
    sns.despine(ax=monitor_ax,left=True,bottom=True)

    plt.tight_layout()

    def init_plot():
        beh_blocker.set(x=0)
        raster_blocker.set(x=0)
        return [beh_blocker,raster_blocker]

    def animate(frame_time):
        beh_blocker.set(x=frame_time)
        raster_blocker.set(x=frame_time)

        frame_idx = int(frame_time/trial['bin_size'])
        hand.set(center=trial['rel_hand_pos'][frame_idx,:2])
        cursor.set(center=trial['rel_cursor_pos'][frame_idx,:2])

        if trial['task']=='RTT':
            idx_targ_start = trial['idx_rtgoCueTimes']
            idx_targ_end = trial['idx_rtHoldTimes']
            on_targs = (idx_targ_start<frame_idx) & (frame_idx<idx_targ_end)
            for target,on_indicator in zip(targets,on_targs):
                target.set(visible=on_indicator)

        if trial['task']=='CST' and frame_idx>trial['idx_cstEndTime']:
                cursor.set(color='y')
    
        return [beh_blocker,raster_blocker]

    frame_interval = 30 #ms
    frames = np.arange(trial['trialtime'][0],trial['trialtime'][-1],frame_interval*1e-3)
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init_plot,
        frames = frames,
        interval = frame_interval,
        blit = True,
    )

    return anim

for trial_to_plot in [227,228]:
    anim = animate_trial_monitor(td.loc[td['trial_id']==trial_to_plot].squeeze())
    anim_name = src.util.format_outfile_name(td,postfix=f'trial_{trial_to_plot}_monitor_anim')
    anim.save(os.path.join('../results/2022_sfn_poster/',anim_name+'.mp4'),writer='ffmpeg',fps=30,dpi=400)
# %%

