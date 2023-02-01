#%%
import numpy as np

import src
import pyaldata
import yaml
import pandas as pd

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit
from src.models import SSA

import seaborn as sns
import matplotlib.pyplot as plt

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
td = (
    src.data.load_clean_data(**load_params)
    .assign(**{'trialtime': lambda x: x['Time from go cue (s)']})
    .pipe(pyaldata.dim_reduce,PCA(n_components=15),'MC_rates','MC_pca')
    .pipe(pyaldata.dim_reduce,PCA(n_components=15),'lfads_rates','lfads_pca')
    .pipe(pyaldata.dim_reduce,SSA(R=15,n_epochs=3000,lr=0.01),'MC_rates','MC_ssa')
    .pipe(pyaldata.dim_reduce,SSA(R=15,n_epochs=3000,lr=0.01),'lfads_rates','lfads_ssa')
)

# %%
trialnum = 228
trial = td.loc[td['trial_id']==trialnum].squeeze()
fig,ax = plt.subplots(4,1,figsize=(10,10))
src.plot.plot_hand_trace(trial,ax=ax[0])
src.plot.plot_hand_velocity(trial,ax=ax[1])
ax[2].plot(trial['lfads_pca'][:,0:3])
ax[3].plot(trial['lfads_inputs'])

# %%
fig,ax = plt.subplots(1,1)
task_colors = {'CST': 'C0','RTT': 'C1'}
signal = 'lfads_pca'
trials_to_plot = [227,228]
for trial_id in trials_to_plot:
    trial = td.loc[td['trial_id']==trial_id].squeeze()
    ax.plot(
        trial[signal][:,0],
        trial[signal][:,1],
        color=task_colors[trial['task']],
    )

sns.despine(ax=ax,left=True,bottom=True)

#%% k3d plots
import k3d
cst_trace_plot = k3d.plot(name='CST smoothed neural traces')
max_abs_hand_vel = np.percentile(np.abs(np.row_stack(td['hand_vel'])[:,0]),95)
# plot traces
for _,trial in td.query('task=="CST"').sample(n=10).iterrows():
    neural_trace = trial['lfads_pca']
    cst_trace_plot+=k3d.line(
        neural_trace[:,0:3].astype(np.float32),
        shader='mesh',
        width=0.5,
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
        width=0.5,
        attribute=trial['hand_vel'][:,0],
        color_map=k3d.paraview_color_maps.Erdc_divHi_purpleGreen,
        color_range=[-max_abs_hand_vel,max_abs_hand_vel],
    )
rtt_trace_plot.display()

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

# %% plot SSA

def plot_trial(trial_to_plot,ax_list,signal_to_plot):
    num_dims = ax_list.shape[0]-1

    beh_ax = ax_list[-1]
    src.plot.plot_hand_trace(trial_to_plot,ax=beh_ax)
    beh_ax.set_ylabel('Pos')

    for i in range(num_dims):
        ax = ax_list[i]
        # Plot SSA results
        ax.plot(trial_to_plot['trialtime'][[0,-1]],[0,0],color='k')
        ax.plot(trial_to_plot['trialtime'],trial_to_plot[signal_to_plot][:,i])
        ax.set_yticks([])
        ax.plot([0,0],ax.get_ylim(),color='k',linestyle='--')
        sns.despine(ax=ax,trim=True)

trials_to_plot = [227,228]
fig,axs = plt.subplots(16,len(trials_to_plot),sharex=True,figsize=(10,10))

for colnum,trial_id in enumerate(trials_to_plot):
    trial = td.loc[td['trial_id']==trial_id].squeeze()
    plot_trial(trial,axs[:,colnum],signal_to_plot='lfads_ssa')


# %% decoding

signal = 'lfads_pca'
def get_test_labels(df):
    gss = GroupShuffleSplit(n_splits=1,test_size=0.25)
    _,test = next(gss.split(
        df['lfads_pca'],
        df['True velocity'],
        groups=df['trial_id'],
    ))
    return np.isin(np.arange(df.shape[0]),test)

def fit_models(df):
    # individual models
    models = {}
    for task in df['task'].unique():
        models[task] = LinearRegression()
        train_df = df.loc[(~df['Test set']) & (df['task']==task)]
        models[task].fit(
            np.row_stack(train_df['lfads_pca']),
            train_df['True velocity'],
        )

    # joint models
    models['Joint'] = LinearRegression()
    train_df = df.loc[~df['Test set']]
    models['Joint'].fit(
        np.row_stack(train_df['lfads_pca']),
        train_df['True velocity'],
    )

    return models

def model_predict(df,models):
    ret_df = df.copy()
    for model_name,model in models.items():
        ret_df = ret_df.assign(**{
            f'{model_name} predicted': model.predict(np.row_stack(ret_df['lfads_pca']))
        })
    return ret_df

def score_models(df,models):
    scores = pd.Series(index=pd.MultiIndex.from_product(
        [df['task'].unique(),models.keys()],
        names=['Test data','Train data']
    ))
    for task in df['task'].unique():
        for model_name, model in models.items():
            test_df = df.loc[df['Test set'] & (df['task']==task)]
            scores[(task,model_name)] = model.score(np.row_stack(test_df['lfads_pca']),test_df['True velocity'])
    
    return scores

td_train_test = (
    td
    .assign(
        **{'True velocity': lambda df: df.apply(lambda s: s['hand_vel'][:,0],axis=1)}
    )
    .filter(items=[
        'trial_id',
        'Time from go cue (s)',
        'task',
        'True velocity',
        'lfads_pca',
    ])
    .explode([
        'Time from go cue (s)',
        'True velocity',
        'lfads_pca',
    ])
    .astype({
        'Time from go cue (s)': float,
        'True velocity': float,
    })
    .assign(**{'Test set': lambda df: get_test_labels(df)})
)

models = fit_models(td_train_test)
scores = score_models(td_train_test,models)
td_pred = (
    td_train_test
    .pipe(model_predict,models)
    .melt(
        id_vars=['trial_id','Time from go cue (s)','task'],
        value_vars=['True velocity','CST predicted','RTT predicted','Joint predicted'],
        var_name='Model',
        value_name='Hand velocity (cm/s)',
    )
)

# trials_to_plot=[71,52]
trials_to_plot=td_pred.groupby('task').sample(n=1)['trial_id']
g=sns.relplot(
    data=td_pred.loc[np.isin(td_pred['trial_id'],trials_to_plot)],
    x='Time from go cue (s)',
    y='Hand velocity (cm/s)',
    hue='Model',
    hue_order=['True velocity','CST predicted','RTT predicted','Joint predicted'],
    palette=['k','C0','C1','0.5'],
    kind='line',
    row='trial_id',
    row_order=trials_to_plot,
    height=4,
    aspect=2,
)
g.axes[0,0].set_yticks([-200,0,200])
g.axes[0,0].set_xticks([0,2,4,6])
sns.despine(fig=g.fig,trim=True)

# fig_name = src.util.format_outfile_name(td,postfix='cst71_rtt52_vel_pred')
# g.fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

fig,ax = plt.subplots(1,1)
sns.heatmap(
    ax=ax,
    data=scores.unstack(),
    vmin=0,
    vmax=1,
    annot=True,
    annot_kws={'fontsize': 21},
    cmap='gray',
)
# fig_name = src.util.format_outfile_name(td,postfix='cst_rtt_vel_pred_scores')
# fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

# %%
