#%%
import src

import pyaldata
import numpy as np
import yaml

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import seaborn as sns
import matplotlib.pyplot as plt
import k3d

with open("../params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

load_params = {
    'file_prefix': 'Prez_20220721',
    'preload_params': params['preload'],
    'chop_merge_params': params['chop_merge'],
    'epoch_fun': src.util.generate_realtime_epoch_fun(
        start_point_name='idx_ctHoldTime',
        end_point_name='idx_endTime',
    ),
}
joint_pca_model = src.models.JointSubspace(n_comps_per_cond=20,signal='lfads_rates',condition='task',remove_latent_offsets=False)
dekodec_model = src.models.DekODec(var_cutoff=0.99,signal='lfads_rates_joint_pca',condition='task')
dekodec_shuffle_model = src.models.DekODec(var_cutoff=0.99,signal='lfads_rates_joint_pca',condition='task_shuffle')
rng = np.random.default_rng()
td = (
    src.data.load_clean_data(**load_params)
    .query('task=="RTT" | task=="CST"')
    .assign(**{
        'trialtime': lambda df: df['Time from go cue (s)'],
        'task_shuffle': lambda df: rng.permutation(df['task'].values),
    })
    .pipe(pyaldata.soft_normalize_signal,signals=['lfads_rates','MC_rates'])
    .pipe(src.data.remove_baseline_rates,signals=['MC_rates','lfads_rates'])
    .pipe(joint_pca_model.fit_transform)
    .pipe(dekodec_model.fit_transform)
)

# %%
td_shuffle = dekodec_shuffle_model.fit_transform(td)

#%% Context space
signal = 'lfads_rates_joint_pca'
td_models = src.data.rebin_data(td,new_bin_size=0.100)

vel_model = LinearRegression(fit_intercept=False)
vel_model.fit(
    np.row_stack(td_models[signal]),
    np.row_stack(td_models['hand_vel'])[:,0],
)

tonic_context_model = LinearDiscriminantAnalysis()
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

td['Motor Cortex Velocity Dim'] = [(sig @ norm_vec(vel_model.coef_).squeeze()[:,None]).squeeze() for sig in td[signal]]
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

trial_fig, score_fig = src.decoder_analysis.run_decoder_analysis(
    td,
    'lfads_rates_joint_pca',
    hand_or_cursor='hand',
    pos_or_vel='acc',
    trace_component=0,
)

# %% plot individual traces

def plot_trial_split_space(trial_to_plot,ax_list):
    src.plot.plot_hand_trace(trial_to_plot,ax=ax_list[0],timesig='Time from go cue (s)')
    src.plot.plot_hand_velocity(trial_to_plot,ax_list[1],timesig='Time from go cue (s)')

    sig_list = [f'{signal}_cst_unique',f'{signal}_rtt_unique',f'{signal}_shared']
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

trials_to_plot = td.groupby('task').sample(n=1).set_index('trial_id')
fig,axs = plt.subplots(32,len(trials_to_plot),sharex=True,sharey='row',figsize=(10,18))
fig.tight_layout()
for colnum,(trial_id,trial) in enumerate(trials_to_plot.iterrows()):
    plot_trial_split_space(trial,axs[:,colnum])

#%% k3d plots
cst_trace_plot = k3d.plot(name='CST smoothed neural traces')
max_abs_hand_vel = np.percentile(np.abs(np.row_stack(td['hand_vel'])[:,0]),95)
# plot traces
for _,trial in td.query('task=="CST"').sample(n=10).iterrows():
    neural_trace = trial['lfads_rates_joint_pca']
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
    neural_trace = trial['lfads_rates_joint_pca']
    rtt_trace_plot+=k3d.line(
        neural_trace[:,0:3].astype(np.float32),
        shader='mesh',
        width=3e-3,
        attribute=trial['hand_vel'][:,0],
        color_map=k3d.paraview_color_maps.Erdc_divHi_purpleGreen,
        color_range=[-max_abs_hand_vel,max_abs_hand_vel],
    )
rtt_trace_plot.display()

#%% Make 2D plots of neural traces in shared and unique spaces
def plot_trial_split_space_2D(trial_to_plot,ax_list,color='k'):
    sig_list = [f'{signal}_shared',f'{signal}_cst_unique',f'{signal}_rtt_unique']
    # sig_list = [f'{signal}_shared',f'{signal}_CST',f'{signal}_RTT']

    for ax,sig in zip(ax_list,sig_list):
        ax.plot(trial_to_plot[sig][:,0],trial_to_plot[sig][:,1],color=color)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax,trim=True)


trials_to_plot = td.groupby('task').sample(n=1).set_index('trial_id')
fig,axs = plt.subplots(3,1,figsize=(4,10))
fig.tight_layout()
for colnum,(trial_id,trial) in enumerate(trials_to_plot.iterrows()):
    plot_trial_split_space_2D(trial,axs,color='C0' if trial['task']=='CST' else 'C1')

fig_name = src.util.format_outfile_name(td,postfix='cst_rtt_split_space_2D')
# fig.savefig(os.path.join('../results/2023_ncm_poster/',fig_name+'.pdf'))
# %% Check out what happens if we project MC_rates into this space we found with LFADS

sig_temp = 'MC_rates'
td_temp = (
    td
    .assign(**{
        f'{sig_temp}_joint_pca': lambda df: df.apply(lambda s: np.dot(s[sig_temp],joint_pca_model.P_),axis=1),
    })
    .assign(**{
        f'{sig_temp}_joint_pca_cst_unique': lambda df: df.apply(lambda s: np.dot(s[f'{sig_temp}_joint_pca'],cst_unique_proj),axis=1),
        f'{sig_temp}_joint_pca_rtt_unique': lambda df: df.apply(lambda s: np.dot(s[f'{sig_temp}_joint_pca'],rtt_unique_proj),axis=1),
        f'{sig_temp}_joint_pca_shared': lambda df: df.apply(lambda s: np.dot(s[f'{sig_temp}_joint_pca'],shared_proj),axis=1),
    })
)


def plot_MC_trial_split_space_2D(trial_to_plot,ax_list,color='k'):
    sig_list = [
        f'{sig_temp}_joint_pca_shared',
        f'{sig_temp}_joint_pca_cst_unique',
        f'{sig_temp}_joint_pca_rtt_unique',
    ]

    for ax,sig in zip(ax_list,sig_list):
        ax.plot(trial_to_plot[sig][:,0],trial_to_plot[sig][:,1],color=color)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax,trim=True)


trials_to_plot = td_temp.groupby('task').sample(n=1).set_index('trial_id')
fig,axs = plt.subplots(3,1,figsize=(4,10))
fig.tight_layout()
for colnum,(trial_id,trial) in enumerate(trials_to_plot.iterrows()):
    plot_MC_trial_split_space_2D(trial,axs,color='C0' if trial['task']=='CST' else 'C1')
# %% Some extra scatter plots

td_explode = (
    td
    .assign(
        **{'Hand velocity (cm/s)': lambda x: x.apply(lambda y: y['hand_vel'][:,0],axis=1)}
    )
    .filter(items=[
        'trial_id',
        'Time from go cue (s)',
        'task',
        'Motor Cortex Velocity Dim',
        'Motor Cortex Transient Context Dim',
        'Motor Cortex Tonic Context Dim',
        'Hand velocity (cm/s)'
    ])
    .explode([
        'Time from go cue (s)',
        'Motor Cortex Velocity Dim',
        'Motor Cortex Transient Context Dim',
        'Motor Cortex Tonic Context Dim',
        'Hand velocity (cm/s)',
    ])
    .astype({
        'Time from go cue (s)': float,
        'Motor Cortex Velocity Dim': float,
        'Motor Cortex Transient Context Dim': float,
        'Motor Cortex Tonic Context Dim': float,
        'Hand velocity (cm/s)': float,
    })
    .loc[lambda df: df['Time from go cue (s)']>0]
    # .loc[lambda df: (df['Time from go cue (s)']<0) & (df['Time from go cue (s)']>-0.5)]
)

vel_corr = (
    td_explode
    .groupby('task')
    .apply(lambda df: np.corrcoef(df['Hand velocity (cm/s)'],df['Motor Cortex Velocity Dim'])[0,1])
)
context_corr = (
    td_explode
    .groupby('task')
    .apply(lambda df: np.corrcoef(df['Hand velocity (cm/s)'],df['Motor Cortex Tonic Context Dim'])[0,1])
)
g = sns.pairplot(
    data=td_explode.sample(300),
    x_vars='Hand velocity (cm/s)',
    y_vars=[
        'Motor Cortex Velocity Dim',
        'Motor Cortex Tonic Context Dim'
    ],
    hue='task',
    hue_order=['CST','RTT'],
    kind='reg',
    height=4,
    aspect=1,
)
sns.despine(fig=g.fig,trim=True)
fig_name = src.util.format_outfile_name(td,postfix='neural_dims_v_vel')

# %% Animate a trial in split subspaces (hand trace, sensorimotor plot, first 2 dims of subspaces)
from ipywidgets import interact
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle,Circle
import matplotlib.animation as animation
import matplotlib as mpl

def animate_trial_timecourse(trial):
    fig = plt.figure(figsize=(10,6))
    gs = mpl.gridspec.GridSpec(2,3,figure=fig)
    beh_ax = fig.add_subplot(gs[0,:2])
    phase_plot_ax = fig.add_subplot(gs[0,2],sharey=beh_ax)
    shared_space_ax = fig.add_subplot(gs[1,0])
    cst_unique_space_ax = fig.add_subplot(gs[1,1])
    rtt_unique_space_ax = fig.add_subplot(gs[1,2])

    src.plot.plot_hand_trace(trial,ax=beh_ax)
    beh_blocker = beh_ax.add_patch(Rectangle((0,-100),10,200,color='w',zorder=100))
    beh_ax.set_xlim([0,6])
    beh_ax.set_xticks([])
    beh_ax.set_xlabel('')
    beh_ax.set_ylabel('Hand position (cm)')
    sns.despine(ax=beh_ax,trim=True,bottom=True)

    phase_plot_ax.plot([-200,200],[0,0],color='k')
    phase_plot_ax.plot([0,0],[-50,50],color='k')
    phase_trace, = phase_plot_ax.plot([],[],color='k')
    phase_plot_ax.set_xlim([-200,200])
    phase_plot_ax.set_xlabel('Hand velocity (cm/s)')
    sns.despine(ax=phase_plot_ax,trim=True)

    shared_space_ax.plot([-0.4,-0.15],[-0.4,-0.4],color='k',lw=5)
    shared_space_ax.text(-0.35,-0.45,'shared comp 1',fontsize=18)
    shared_space_ax.plot([-0.4,-0.4],[-0.4,-0.15],color='k',lw=5)
    shared_space_ax.text(-0.475,-0.3,'shared comp 2',fontsize=18,rotation=90)
    shared_space_trace, = shared_space_ax.plot([],[],color='k')
    shared_space_ax.set_xlim([-0.5,0.5])
    shared_space_ax.set_ylim([-0.5,0.5])
    shared_space_ax.set_xticks([])
    shared_space_ax.set_yticks([])
    sns.despine(ax=shared_space_ax,left=True,bottom=True)

    cst_unique_space_ax.plot([-0.4,-0.15],[-0.4,-0.4],color='k',lw=5)
    cst_unique_space_ax.text(-0.35,-0.45,'cst unique comp 1',fontsize=18)
    cst_unique_space_ax.plot([-0.4,-0.4],[-0.4,-0.15],color='k',lw=5)
    cst_unique_space_ax.text(-0.475,-0.3,'cst unique comp 2',fontsize=18,rotation=90)
    cst_unique_space_trace, = cst_unique_space_ax.plot([],[],color='k')
    cst_unique_space_ax.set_xlim([-0.5,0.5])
    cst_unique_space_ax.set_ylim([-0.5,0.5])
    cst_unique_space_ax.set_xticks([])
    cst_unique_space_ax.set_yticks([])
    sns.despine(ax=cst_unique_space_ax,left=True,bottom=True)

    rtt_unique_space_ax.plot([-0.4,-0.15],[-0.4,-0.4],color='k',lw=5)
    rtt_unique_space_ax.text(-0.35,-0.45,'rtt unique comp 1',fontsize=18)
    rtt_unique_space_ax.plot([-0.4,-0.4],[-0.4,-0.15],color='k',lw=5)
    rtt_unique_space_ax.text(-0.475,-0.3,'rtt unique comp 2',fontsize=18,rotation=90)
    rtt_unique_space_trace, = rtt_unique_space_ax.plot([],[],color='k')
    rtt_unique_space_ax.set_xlim([-0.5,0.5])
    rtt_unique_space_ax.set_ylim([-0.5,0.5])
    rtt_unique_space_ax.set_xticks([])
    rtt_unique_space_ax.set_yticks([])
    sns.despine(ax=rtt_unique_space_ax,left=True,bottom=True)

    plt.tight_layout()

    def plot_trial_timecourse(trial,end_idx=None):
        beh_blocker.set(x=trial['trialtime'][end_idx])

        phase_trace.set_data(
            trial['hand_vel'][:end_idx,0],
            trial['rel_hand_pos'][:end_idx,0],
        )
    
        # first two PCs
        shared_space_trace.set_data(
            trial['lfads_rates_joint_pca_shared'][:end_idx,0],
            trial['lfads_rates_joint_pca_shared'][:end_idx,1],
        )
        cst_unique_space_trace.set_data(
            trial['lfads_rates_joint_pca_cst_unique'][:end_idx,0],
            trial['lfads_rates_joint_pca_cst_unique'][:end_idx,1],
        )
        rtt_unique_space_trace.set_data(
            trial['lfads_rates_joint_pca_rtt_unique'][:end_idx,0],
            trial['lfads_rates_joint_pca_rtt_unique'][:end_idx,1],
        )
        return [beh_blocker,phase_trace,shared_space_trace,cst_unique_space_trace,rtt_unique_space_trace]

    def init_plot():
        beh_blocker.set(x=0)
        phase_trace.set_data([],[])
        shared_space_trace.set_data([],[])
        cst_unique_space_trace.set_data([],[])
        rtt_unique_space_trace.set_data([],[])
        return [beh_blocker,phase_trace,shared_space_trace,cst_unique_space_trace,rtt_unique_space_trace]

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

trials_to_plot = td.groupby('task')['trial_id'].sample(n=3,random_state=0).values
# trials_to_plot=[227,228]
for trial_to_plot in trials_to_plot:
    anim = animate_trial_timecourse(td.loc[td['trial_id']==trial_to_plot].squeeze())
    anim_name = src.util.format_outfile_name(td,postfix=f'trial_{trial_to_plot}_anim')
    anim.save(os.path.join('../results/20230814_smile_meeting/',anim_name+'.mp4'),writer='ffmpeg',fps=15,dpi=400)
# %% split subspace variance plots

def plot_split_subspace_variance(td,signal='lfads_rates_joint_pca'):
    def calculate_percent_variance(arr,col):
        return np.var(arr[:,col])/np.var(arr,axis=0).sum()

    compared_var = (
        td
        .groupby('task')
        [[f'{signal}',f'{signal}_split']]
        .agg([
            lambda s,col=col: calculate_percent_variance(np.row_stack(s),col)
            for col in range(40)
        ])
        .rename({'lfads_rates_joint_pca': 'unsplit','lfads_rates_joint_pca_split': 'split'},axis=1,level=0)
        .rename(lambda label: label.strip('<lambda_>'),axis=1,level=1)
        .unstack()
        .reset_index()
        .rename({
            'level_0': 'neural space',
            'level_1':'component',
            0: 'fraction variance'
        },axis=1)
    )

    sns.catplot(
        data=compared_var,
        x='component',
        y='fraction variance',
        hue='task',
        kind='bar',
        row='neural space',
        sharex=True,
        sharey=True,
        aspect=2,
        height=3,
    )

plot_split_subspace_variance(td,signal='lfads_rates_joint_pca')
plot_split_subspace_variance(td_shuffle,signal='lfads_rates_joint_pca')


# fig,axs = plt.subplots(2,1,sharex=True,sharey=True,figsize=(6,6))
# sns.barplot(
#     ax=axs[0],
#     data=unsplit_var,
#     x='component',
#     y='fraction variance',
#     hue='task',
# )
# sns.barplot(
#     ax=axs[1],
#     data=split_var,
#     x='component',
#     y='fraction variance',
#     hue='task',
# )
# sns.despine(fig=fig,trim=True)

# %%
td_subspace_overlap = (
    td
    .pipe(src.data.rebin_data,new_bin_size=0.05)
    .groupby('task')
    ['lfads_rates']
    .pipe(src.subspace_tools.bootstrap_subspace_overlap,num_bootstraps=10,var_cutoff=0.99)
    .filter(items=['task_data','task_proj','boot_id','subspace_overlap','subspace_overlap_rand'])
    .assign(within_task=(lambda x: x['task_proj']==x['task_data']))
    .melt(
        id_vars=['within_task','boot_id'],
        value_vars=['subspace_overlap','subspace_overlap_rand'],
        value_name='Subspace overlap',
        var_name='is_control',
    )
    .assign(is_control=lambda x: x['is_control']=='subspace_overlap_rand')
    .assign(Category= lambda s: np.where(s['is_control'],'Control',np.where(s['within_task'],'Within','Across')))
)
fig,ax = plt.subplots(1,1)
sns.barplot(
    ax=ax,
    data=td_subspace_overlap,
    x='Subspace overlap',
    y='Category',
    color='0.7',
)
sns.despine(ax=ax,trim=True)
plt.tight_layout()
# %%
