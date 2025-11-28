import yaml
import os
import pandas as pd
# Load run configuration and universe list.
params = yaml.safe_load(open("../../conf/params.yaml"))

# Unpack relevant parameters for feature calculation.
prediction_periods = params['DATA_PREP']['PREDICTION_PERIODS']
ema_periods = params['DATA_PREP']['EMA_PERIODS']
slope_periods = params['DATA_PREP']['SLOPE_PERIODS']
z_norm_window = params['DATA_PREP']['Z_NORM_WINDOW']


# Unpack date boundaries for train/validation/test splits.
train_date = params['DATA_PREP']['TRAIN_DATE']
validation_date = params['DATA_PREP']['VALIDATION_DATE']
test_date = params['DATA_PREP']['TEST_DATE']

# Start processing from a given offset within the symbol list (useful for chunking runs).
counter = 76

