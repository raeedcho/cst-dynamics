from lfads_tf2.utils import load_data, load_posterior_averages
import src.data
import pyaldata
import yaml

with open("params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)["lfads_prep"]

trial_data = src.data.load_clean_data("data/trial_data/Earl_20190716_COCST_TD.mat")
td_co = trial_data.groupby("task").get_group("CO")
td_co = pyaldata.combine_time_bins(
    td_co, int(params["bin_size"] / td_co["bin_size"].values[0])
)

trial_ids = load_data(
    "data/pre-lfads/", prefix="Earl_20190716", signal="trial_id", merge_tv=True
)[0].astype(int)

post_data = load_posterior_averages("results/lfads/Earl_20190716_CO", merge_tv=True)

# lfads_trial_rates_dict = src.data.merge_tensor_into_trials(post_data.rates, trial_ids, overlap=80)

td_co = src.data.add_lfads_rates(
    td_co, post_data.rates, chopped_trial_ids=trial_ids, overlap=params["overlap"]
)

td_co = src.data.trim_nans(td_co, ref_signals=["rel_hand_pos", "lfads_rates"])

