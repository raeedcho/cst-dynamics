#%%
from lfads_tf2.utils import load_data, load_posterior_averages
import src
import pyaldata
import yaml
import seaborn as sns

with open("../params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)["lfads_prep"]

load_params = {
    'file_prefix': 'Prez_20220721',
    'verbose': False,
    'keep_unsorted': False,
    # 'bin_size': params['bin_size'],
    'bin_size': 0.01,
    'lfads_params': params,
}
trial_data = src.data.load_clean_data(**load_params)

#%%
