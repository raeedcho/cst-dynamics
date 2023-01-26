#%%
from lfads_tf2.utils import load_data, load_posterior_averages
import src
import pyaldata
import yaml

import seaborn as sns
import matplotlib.pyplot as plt

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

# %%
trial_data['trialtime'] = trial_data['Time from go cue (s)']

trialnum = 15
fig,ax = plt.subplots(3,1)
src.plot.plot_hand_trace(trial_data.loc[trialnum],ax=ax[0])
ax[1].plot(trial_data.loc[trialnum]['lfads_rates'][:,0:15])
ax[2].plot(trial_data.loc[trialnum]['lfads_inputs'])

# %%
