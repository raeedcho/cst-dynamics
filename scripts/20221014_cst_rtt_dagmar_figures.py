'''
Dagmar's giving a 20-ish minute talk about CST work--she'll mainly be focusing on Mohsen's optimal feedback control modeling work with human subjects, but wants to throw in some teasers on trying to understand neural activity at the end. She estimates maybe 3-5 minutes devoted to it, so it needs to be short and sweet.

Here's an outline of what I think should go into that 3-5 minutes:

- A visual description of the RTT task as a comparison to CST (which will already have been described by this point)
- A representation of the behavioral traces from CST and RTT--basically a few examples of hand position against time for both tasks. Dagmar is also asking for a cursor v. time trace on the CST plots for visual continuity, but RTT can just have the hand position.
- A plot showing neural activity along the axis that best correlates with hand velocity, plotted against hand velocity
- A plot showing neural activity along the axis that separates context, plotted against hand velocity.

And that's probably all that can really fit in there I think.

I suppose I'll do the RTT figure in Illustrator, but I'll generate the rest here.
'''

#%% Setup
import src
import pyaldata
import pandas as pd
import numpy as np
import yaml
import os

from sklearn.decomposition import PCA

from ipywidgets import interact
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle,Circle
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
sns.set_context('talk')

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
    .pipe(pyaldata.restrict_to_interval,start_point_name='idx_goCueTime',end_point_name='idx_endTime',rel_end=-1,warn_per_trial=True)
    .pipe(src.data.add_trial_time)
)

#%% Fit velocity and context models
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

vel_model = LinearRegression(fit_intercept=False)
context_model = LinearDiscriminantAnalysis()

td_models = src.data.rebin_data(td,new_bin_size=0.100)

vel_model.fit(
    np.row_stack(td_models['MC_pca']),
    np.row_stack(td_models['hand_vel'])[:,0],
)

context_model.fit(
    np.row_stack(td_models.apply(lambda x: x['MC_pca'][20,:],axis=1)),
    td_models['task'],
)

def norm_vec(vec):
    return vec/np.linalg.norm(vec)

td['Motor Cortex Velocity Dim'] = [(sig @ norm_vec(vel_model.coef_).squeeze()[:,None]).squeeze() for sig in td['MC_pca']]
td['Motor Cortex Context Dim'] = [(sig @ norm_vec(context_model.coef_).squeeze()[:,None]).squeeze() for sig in td['MC_pca']]

print(f'Angle between velocity dim and context dim: {src.util.angle_between(vel_model.coef_.squeeze(),context_model.coef_.squeeze())} degrees')

#%% Behavioral traces
def plot_trial(trial):
    targ_size = 10
    fig,axs=plt.subplots(3,1,sharex=True,figsize=(6,6))
    
    src.plot.plot_hand_trace(trial,ax=axs[0])

    # hand velocity
    axs[1].plot([0,trial['trialtime'][-1]],[0,0],'-k')
    axs[1].plot(
        trial['trialtime'],
        trial['hand_vel'][:,0],
        color='r',
    )
    axs[1].set_ylim([-300,300])

    # neural dimension
    axs[2].plot(
        trial['trialtime'],
        trial['Motor Cortex Velocity Dim'],
        color=[0.5,0.5,0.5],
    )
    axs[2].set_yticks([])

    axs[0].set_xlabel('')
    axs[0].set_ylabel('Hand position')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Hand velocity')
    axs[2].set_xlabel('Time from go cue (s)')
    axs[2].set_ylabel('Motor cortex\nvelocity dim')
    axs[0].set_title(f'Trial {trial["trial_id"]} ({trial["task"]})')

    sns.despine(fig=fig,trim=True)
    return fig

fig_name = src.util.format_outfile_name(td,postfix='trial_traces')
with PdfPages(os.path.join('../results/dagmar_talk/',fig_name+'.pdf')) as pdf:
    for _,trial in td.iterrows():
        fig = plot_trial(trial)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

#%% Neural dimensions
td_explode = (
    td
    .assign(
        **{'Hand velocity (cm/s)': lambda x: x.apply(lambda y: y['hand_vel'][:,0],axis=1)}
    )
    .loc[:,['trial_id','trialtime','task','Motor Cortex Velocity Dim','Motor Cortex Context Dim','Hand velocity (cm/s)']]
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
g.fig.savefig(os.path.join('../results/dagmar_talk/',fig_name+'.pdf'))

#%%
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
fig.savefig(os.path.join('../results/dagmar_talk/',fig_name+'.pdf'))

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
anim.save(os.path.join('../results/dagmar_talk/',anim_name+'.mp4'),writer='ffmpeg',fps=30,dpi=400)

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
anim.save(os.path.join('../results/dagmar_talk/',anim_name+'.mp4'),writer='ffmpeg',fps=30,dpi=400)

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
    anim.save(os.path.join('../results/dagmar_talk/',anim_name+'.mp4'),writer='ffmpeg',fps=30,dpi=400)
# %%
