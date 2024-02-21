import src
import pyaldata
import yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def main():
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

    td = (
        src.data.load_clean_data(**load_params)
        .query('task=="RTT" | task=="CST"')
        .assign(**{
            'trialtime': lambda df: df['Time from go cue (s)'],
        })
        .pipe(pyaldata.soft_normalize_signal,signals=['lfads_rates','MC_rates'])
        .pipe(src.data.remove_baseline_rates,signals=['MC_rates','lfads_rates'])
    )

    if params['subspace_split']['train_size'] < 1:
        td_train, td_test = train_test_split(td,train_size=params['subspace_split']['train_size'],stratify=td['task'])
    elif params['subspace_split']['train_size'] == 1:
        td_train = td
        td_test = td
    
    subspace_split_pipeline = Pipeline([
        ('joint_pca',src.models.JointSubspace(n_comps_per_cond=20,signal='lfads_rates',condition='task',remove_latent_offsets=False)),
        ('dekodec',src.models.DekODec(var_cutoff=0.99,signal='lfads_rates_joint_pca',condition='task')),
    ])

    td_train = subspace_split_pipeline.fit_transform(td_train)
    td_test = subspace_split_pipeline.transform(td_test)

    td_test.to_pickle('../results/dekodec/Prez_20220721_dekodec_split_test.pkl')

if __name__=='__main__':
    main()