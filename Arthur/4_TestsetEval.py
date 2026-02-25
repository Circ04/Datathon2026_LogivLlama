#%%
## _________________Final Evaluation RF on Test set_____________________________
## Do logged AUC on full test set. Since IPS uses exploding observations, 
# evaluation isn't possible on full test, so take a random 500k subset
import polars as pl
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
 

#%% 
TRAIN_PATH = "train_preprocessed_100k.parquet"
VAL_PATH   = "val_preprocessed_100k.parquet"

# If you've written a single giant parquet:
TEST_FULL_PARQUET = "test.parquet"

#%%

train = pl.read_parquet(TRAIN_PATH)
val   = pl.read_parquet(VAL_PATH)
test  = pl.read_parquet(TEST_FULL_PARQUET)

# %%
