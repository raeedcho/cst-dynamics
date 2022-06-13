"""
This script tries to train LFADS on some Center-Out data
from Earl_20190716.

The main steps to running LFADS are:

    1. Create a configuration YAML file to overwrite any or
        all of the defaults
    2. Create an LFADS object by passing the path to the 
        configuration file.
    3. Train the model using `model.train`
    4. Perform posterior sampling to create the posterior 
        sampling file using `model.sample_and_average`
    5. Load rates, etc. for further processing using 
        `lfads_tf2.utils.load_posterior_averages`

"""

from lfads_tf2.utils import restrict_gpu_usage

restrict_gpu_usage(gpu_ix=0)
from os import path

from lfads_tf2.models import LFADS

# create and train the LFADS model
cfg_path = path.join("configs", "lfads_config_Earl_20190716_CO_10ms.yaml")
model = LFADS(cfg_path=cfg_path)
model.train()
model.sample_and_average()
