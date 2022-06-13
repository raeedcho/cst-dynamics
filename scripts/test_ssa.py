import src.data
import src.util
import yaml
import pyaldata
from sklearn.decomposition import PCA
import numpy as np
from src.models import SSA

trial_data = src.data.load_clean_data("data/trial_data/Earl_20190716_COCST_TD.mat")

with open("params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)["lfads_prep"]

td_cst = trial_data.groupby("task").get_group("CST")
td_cst = pyaldata.combine_time_bins(
    td_cst, int(params["bin_size"] / td_cst["bin_size"].values[0])
)

td_cst["M1_rates"] = [
    pyaldata.smooth_data(spikes / bin_size, dt=bin_size, std=0.05, backend="convolve",)
    for spikes, bin_size in zip(td_cst["M1_spikes"], td_cst["bin_size"])
]
M1_pca_model = PCA()
td_cst = pyaldata.dim_reduce(td_cst, M1_pca_model, "M1_rates", "M1_pca")

start_time = -0.4
end_time = 5.0
td_cst = pyaldata.restrict_to_interval(
    td_cst,
    start_point_name="idx_goCueTime",
    rel_start=int(start_time / td_cst.loc[td_cst.index[0], "bin_size"]),
    rel_end=int(end_time / td_cst.loc[td_cst.index[0], "bin_size"]),
    reset_index=False,
)

num_dims = 10
M1_ssa_model = SSA(R=num_dims, n_epochs=3000, lr=0.01)
td_cst = pyaldata.dim_reduce(td_cst, M1_ssa_model, "M1_rates", "M1_ssa")
