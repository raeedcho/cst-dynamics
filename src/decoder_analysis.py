import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

def get_test_labels(df,signal='MC_rates'):
    gss = GroupShuffleSplit(n_splits=1,test_size=0.25)
    _,test = next(gss.split(
        df['True velocity'],
        groups=df['trial_id'],
    ))
    return np.isin(np.arange(df.shape[0]),test)

def fit_models(df,signal):
    subsampled_training_df = (
        df
        .groupby('Test set')
        .get_group(False)
        .groupby('task')
        .sample(n=60000)
    )
    # individual models
    models = {}
    for task in df['task'].unique():
        models[task] = LinearRegression()
        train_df = subsampled_training_df.groupby('task').get_group(task)
        models[task].fit(
            np.row_stack(train_df[signal]),
            train_df['True velocity'],
        )

    # joint models
    models['Joint'] = LinearRegression()
    train_df = df.loc[~df['Test set']].groupby('task').sample(n=30000)
    models['Joint'].fit(
        np.row_stack(train_df[signal]),
        train_df['True velocity'],
    )

    return models

def model_predict(df,signal,models):
    ret_df = df.copy()
    for model_name,model in models.items():
        ret_df = ret_df.assign(**{
            f'{model_name} predicted': model.predict(np.row_stack(ret_df[signal]))
        })
    return ret_df

def score_models(df,signal,models):
    scores = pd.Series(index=pd.MultiIndex.from_product(
        [df['task'].unique(),models.keys()],
        names=['Test data','Train data']
    ))
    for task in df['task'].unique():
        for model_name, model in models.items():
            test_df = df.loc[df['Test set'] & (df['task']==task)]
            scores[(task,model_name)] = model.score(np.row_stack(test_df[signal]),test_df['True velocity'])
    
    return scores
    
def run_decoder_analysis(td,signal):
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
            signal,
        ])
        .explode([
            'Time from go cue (s)',
            'True velocity',
            signal,
        ])
        .astype({
            'Time from go cue (s)': float,
            'True velocity': float,
        })
        .assign(**{'Test set': lambda df: get_test_labels(df)})
    )
    
    models = fit_models(td_train_test,signal)
    scores = score_models(td_train_test,signal,models)
    td_pred = (
        td_train_test
        .pipe(model_predict,signal,models)
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
    
    heatmap_fig,ax = plt.subplots(1,1)
    sns.heatmap(
        ax=ax,
        data=scores.unstack(),
        vmin=0,
        vmax=1,
        annot=True,
        annot_kws={'fontsize': 21},
        cmap='gray',
    )

    return g.fig, heatmap_fig
