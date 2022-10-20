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
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
sns.set_context('talk')

#%load_ext autoreload
#%autoreload 2

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
    fig,axs=plt.subplots(3,1,sharex=True,figsize=(6,6))
    # hand position
    axs[0].plot([0,trial['trialtime'][-1]],[0,0],'-k')
    axs[0].plot(
        trial['trialtime'],
        trial['rel_cursor_pos'][:,0],
        c='b',
        # alpha=0.25,
    )
    axs[0].plot(
        trial['trialtime'],
        trial['rel_hand_pos'][:,0],
        c='r',
    )
    axs[0].set_ylim(-60,60)

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

    axs[0].set_ylabel('Hand position')
    axs[1].set_ylabel('Hand velocity')
    axs[2].set_ylabel('Motor cortex\nvelocity dim')
    axs[2].set_xlabel('Time from go cue (s)')
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

# %%
fig = plt.figure(figsize=(10,6))
gs = mpl.gridspec.GridSpec(2,2,figure=fig)
beh_ax = fig.add_subplot(gs[0,0])
raster_ax = fig.add_subplot(gs[1,0])
pop_ax = fig.add_subplot(gs[:,1])

beh_ax.plot([0,6],[0,0],color='k')
cursor_trace, = beh_ax.plot([],[],color='b')
hand_trace, = beh_ax.plot([],[],color='r')
beh_ax.set_xlim([0,6])
beh_ax.set_ylim([-50,50])
beh_ax.set_xticks([])
beh_ax.set_ylabel('Hand position (cm)')
sns.despine(ax=beh_ax,trim=True,bottom=True)

raster_trace, =  raster_ax.plot([],[],'|k',markersize=1)
raster_ax.set_ylim([0,88])
raster_ax.set_xlim([0,6])
raster_ax.set_yticks([])
raster_ax.set_ylabel('Neurons')
raster_ax.set_xlabel('Time from go cue (s)')
sns.despine(ax=raster_ax, left=True, bottom=False, trim=True, offset=10)

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

def plot_trial_timecourse(trial,slicer=slice(None)):
    cursor_trace.set_data(
        trial['trialtime'][slicer],
        trial['rel_cursor_pos'][slicer,0],
    )
    hand_trace.set_data(
        trial['trialtime'][slicer],
        trial['rel_hand_pos'][slicer,0],
    )

    # raster
    spike_bins,spike_neurons = np.nonzero(trial['MC_spikes'][slicer,:])
    raster_trace.set_data(
        trial['bin_size']*spike_bins,
        spike_neurons,
    )

    # first two PCs
    pop_trace.set_data(
        trial['MC_pca'][slicer,0],
        trial['MC_pca'][slicer,1],
    )
    return [cursor_trace,hand_trace,raster_trace,pop_trace]

    # fig.suptitle(f'Trial {trial["trial_id"]} ({trial["task"]})')


def animate_trial_timecourse(trial):
    def init_plot():
        cursor_trace.set_data([],[])
        hand_trace.set_data([],[])
        raster_trace.set_data([],[])
        pop_trace.set_data([],[])
        return [cursor_trace,hand_trace,raster_trace,pop_trace]

    def animate(frame_time):
        epoch_fun = src.util.generate_realtime_epoch_fun(
            start_point_name='idx_goCueTime',
            rel_end_time=frame_time,
        )
        anim_slice = epoch_fun(trial)

        return plot_trial_timecourse(trial,slicer=anim_slice)

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
# plot_trial_timecourse(td.loc[td['trial_id']==trial_to_plot].squeeze())
anim = animate_trial_timecourse(td.loc[td['trial_id']==trial_to_plot].squeeze())
anim_name = src.util.format_outfile_name(td,postfix=f'trial_{trial_to_plot}_anim')
anim.save(os.path.join('../results/dagmar_talk/',anim_name+'.mp4'),writer='ffmpeg',fps=30,dpi=400)

# %%
def animate_context_vel_dims(cst_trial,rtt_trial):
    fig = plt.figure(figsize=(10,6))
    gs = mpl.gridspec.GridSpec(2,2,figure=fig)
    rtt_ax = fig.add_subplot(gs[0,0])
    cst_ax = fig.add_subplot(gs[1,0])
    pop_ax = fig.add_subplot(gs[:,1])

    rtt_ax.plot([0,6],[0,0],color='k')
    rtt_hand_trace, = rtt_ax.plot([],[],color='r')
    rtt_ax.set_xlim([0,6])
    rtt_ax.set_ylim([-50,50])
    rtt_ax.set_xticks([])
    rtt_ax.set_ylabel('RTT\nHand position (cm)')
    sns.despine(ax=rtt_ax,trim=True,bottom=True)

    cst_ax.plot([0,6],[0,0],color='k')
    cst_cursor_trace, = cst_ax.plot([],[],color='b')
    cst_hand_trace, = cst_ax.plot([],[],color='r')
    cst_ax.set_xlim([0,6])
    cst_ax.set_ylim([-50,50])
    cst_ax.set_xticks([])
    cst_ax.set_ylabel('CST\nHand position (cm)')
    sns.despine(ax=cst_ax,trim=True,bottom=True)

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
        rtt_hand_trace.set_data([],[])
        cst_hand_trace.set_data([],[])
        cst_cursor_trace.set_data([],[])
        cst_pop_trace.set_data([],[])
        rtt_pop_trace.set_data([],[])

        return [rtt_hand_trace,cst_hand_trace,cst_cursor_trace,cst_pop_trace,rtt_pop_trace]

    def animate(frame_time):
        epoch_fun = src.util.generate_realtime_epoch_fun(
            start_point_name='idx_goCueTime',
            rel_end_time=frame_time,
        )
        cst_anim_slice = epoch_fun(cst_trial)
        rtt_anim_slice = epoch_fun(rtt_trial)

        rtt_hand_trace.set_data(
            rtt_trial['trialtime'][rtt_anim_slice],
            rtt_trial['rel_hand_pos'][rtt_anim_slice,0],
        )
        cst_hand_trace.set_data(
            cst_trial['trialtime'][cst_anim_slice],
            cst_trial['rel_hand_pos'][cst_anim_slice,0],
        )
        cst_cursor_trace.set_data(
            cst_trial['trialtime'][cst_anim_slice],
            cst_trial['rel_cursor_pos'][cst_anim_slice,0],
        )
        cst_pop_trace.set_data(
            cst_trial['Motor Cortex Velocity Dim'][cst_anim_slice],
            cst_trial['Motor Cortex Context Dim'][cst_anim_slice],
        )
        rtt_pop_trace.set_data(
            rtt_trial['Motor Cortex Velocity Dim'][rtt_anim_slice],
            rtt_trial['Motor Cortex Context Dim'][rtt_anim_slice],
        )

        return [rtt_hand_trace,cst_hand_trace,cst_cursor_trace,cst_pop_trace,rtt_pop_trace]

    frame_interval = 30 #ms
    frames = np.arange(0,5,frame_interval*1e-3)
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

# %%
