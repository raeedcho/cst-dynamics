import src
import pyaldata
import yaml

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

    joint_pca_model = src.models.JointSubspace(n_comps_per_cond=20,signal='lfads_rates',condition='task',remove_latent_offsets=False)
    dekodec_model = src.models.DekODec(var_cutoff=0.99,signal='lfads_rates_joint_pca',condition='task')
    td = (
        src.data.load_clean_data(**load_params)
        .query('task=="RTT" | task=="CST"')
        .assign(**{
            'trialtime': lambda df: df['Time from go cue (s)'],
        })
        .pipe(pyaldata.soft_normalize_signal,signals=['lfads_rates','MC_rates'])
        .pipe(src.data.remove_baseline_rates,signals=['MC_rates','lfads_rates'])
        .pipe(joint_pca_model.fit_transform)
        .pipe(dekodec_model.fit_transform)
    )

    td.to_pickle('../results/dekodec/Prez_20220721_dekodec_split.pkl')

if __name__=='__main__':
    main()