"""
This module loads a trial data file and associated LFADS data
and puts them together into a single output file.
"""

#%%
import src.data
import yaml

with open("../params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)


#%%
def main():
    load_params = {
        'file_prefix': 'Prez_20220721',
        'preload_params': params['preload'],
        'chop_merge_params': params['chop_merge'],
        'epoch_fun': src.util.generate_realtime_epoch_fun(
            start_point_name='idx_ctHoldTime',
            end_point_name='idx_endTime',
        ),
    }
    # td = src.data.load_clean_data(**load_params)
    td = src.data.preload_data(file_prefix=load_params['file_prefix'],**load_params['preload_params'])
    # td.to_pickle('../results/temp_td.pkl')

if __name__=='__main__':
    main()

# %%
