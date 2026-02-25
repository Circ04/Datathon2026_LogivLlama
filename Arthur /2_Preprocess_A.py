#%% context: We are building a reward model to estimate the probability 
# of session completion given context and template choice. We then use 
# this model as a greedy policy to select the template with highest 
# predicted reward from the eligible pool. Since we only observe 
# rewards for the logged template, we evaluate the new policy using 
# inverse propensity scoring (IPS), assuming the logging policy was 
# uniform random within pools.

import polars as pl
import numpy as np

#%% ---- Load data (SAMPLES) ----
TRAIN_PATH = "train_rsample_100k.parquet"  # your good sample
VAL_PATH   = "val_rsample_100k.parquet"

train = pl.read_parquet(TRAIN_PATH)
val   = pl.read_parquet(VAL_PATH)

print("Train:", train.shape)
print("Val:", val.shape)
print("Train baseline:", train.select(pl.mean("session_end_completed")).item())
print("Val baseline:", val.select(pl.mean("session_end_completed")).item())

#%% ---- Basic EDA and processing ----
train.head(10)

#%% 
# fix hour_UTC (not continuous 0 to 1)
train = train.with_columns(
    ((pl.col("datetime") % 1) * 24)
        .floor()
        .cast(pl.Int8)
        .alias("hour_utc")
)

val = val.with_columns(
    ((pl.col("datetime") % 1) * 24)
        .floor()
        .cast(pl.Int8)
        .alias("hour_utc")
)


#%% Turn UI language into categories (one hot encoding)
#! Don't forget to drop one for Logistic regression (perfect multicollinearity)
#%% 23 lang 
train.select("ui_language").unique()
#%% 23 lang ok 
val.select("ui_language").unique()
#%% 
train = train.to_dummies(columns=["ui_language"])
val   = val.to_dummies(columns=["ui_language"])
#%%
print(len(train.columns))
train.head(10)

#%% Check if one hot encoded have same column names and 
# same order in both val and train
set(train.columns) == set(val.columns)

assert train.columns == val.columns



## To drop for modeling: history

## history_length: number of past exposures
## n_eligible: number of templates a user is able to receive at the moment


#%% 
# **time_since_last_notification**: The number of days since the user received 
#                       their most recent notification (any template).


train = train.with_columns(
    pl.col("history").map_elements(
        lambda hist: 100.0 if (hist is None or len(hist) == 0)
        else float(min(item["n_days"] for item in hist)),
        return_dtype=pl.Float64
    ).alias("time_since_last_notification_days")
)

val = val.with_columns(
    pl.col("history").map_elements(
        lambda hist: 100.0 if (hist is None or len(hist) == 0)
        else float(min(item["n_days"] for item in hist)),
        return_dtype=pl.Float64
    ).alias("time_since_last_notification_days")
)
#%% 
# **Template_freq_last_5**: The max count of any template in the last 5

from collections import Counter

train = train.with_columns(
    pl.col("history").map_elements(
        lambda hist: 0 if (hist is None or len(hist) == 0)
        else max(
            Counter(
                # take 5 most recent by smallest n_days
                [item["template"] for item in sorted(hist, key=lambda d: d["n_days"])[:5]]
            ).values()
        ),
        return_dtype=pl.Int16
    ).alias("max_template_count_last_5")
)

val = val.with_columns(
    pl.col("history").map_elements(
        lambda hist: 0 if (hist is None or len(hist) == 0)
        else max(
            Counter(
                [item["template"] for item in sorted(hist, key=lambda d: d["n_days"])[:5]]
            ).values()
        ),
        return_dtype=pl.Int16
    ).alias("max_template_count_last_5")
)


#%% 
# Recency (Added from Alexander): days since *selected_template* was last shown (min n_days for matching template)
def _days_since_shown(row):
    tmpl = row["selected_template"]
    hist = row["history"]
    if hist is None or len(hist) == 0:
        return 999.0
    matching = [float(h["n_days"]) for h in hist if h["template"] == tmpl]
    return float(min(matching)) if matching else 999.0

train = train.with_columns(
    pl.struct(["selected_template", "history"])
      .map_elements(_days_since_shown, return_dtype=pl.Float64)
      .alias("days_since_shown")
)
val = val.with_columns(
    pl.struct(["selected_template", "history"])
      .map_elements(_days_since_shown, return_dtype=pl.Float64)
      .alias("days_since_shown")
)

#%% 
## Add day index 
train = train.with_columns(
    (pl.col("datetime").floor().cast(pl.Int32) + 1).alias("day_index")
)

val = val.with_columns(
    (pl.col("datetime").floor().cast(pl.Int32) + 1).alias("day_index")
)

#%% remove history
train = train.drop(["history"])
val   = val.drop(["history"])



# %%
OUT_TRAIN = "train_preprocessed_100k.parquet"
OUT_VAL   = "val_preprocessed_100k.parquet"

train.write_parquet(OUT_TRAIN)
val.write_parquet(OUT_VAL)

print("Wrote:", OUT_TRAIN, OUT_VAL)
# %%
