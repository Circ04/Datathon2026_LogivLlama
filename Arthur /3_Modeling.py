#%%
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

#%%
TRAIN_PATH = "train_preprocessed_100k.parquet"
VAL_PATH   = "val_preprocessed_100k.parquet"

train = pl.read_parquet(TRAIN_PATH)
val   = pl.read_parquet(VAL_PATH)

print(train.shape, val.shape)


#%% Recency helper (Added from Alexander): days since *selected_template* was last shown (min n_days for matching template)
SENTINEL_RECENCY = 100.0

def _days_since_shown(row) -> float:
    tmpl = row["selected_template"]
    hist = row["history"]
    if hist is None or len(hist) == 0:
        return SENTINEL_RECENCY
    matching = [float(h["n_days"]) for h in hist if h.get("template") == tmpl]
    return float(min(matching)) if matching else SENTINEL_RECENCY

def add_recency(df: pl.DataFrame, out_col: str = "recency") -> pl.DataFrame:
    return df.with_columns(
        pl.struct(["selected_template", "history"])
          .map_elements(_days_since_shown, return_dtype=pl.Float64)
          .alias(out_col)
    )

#%% Sanity check (should print [])
needed = [
    "session_end_completed",
    "selected_template",
    "eligible_templates",
    "history", # required for recency
    "n_eligible",
    "history_length",
    "hour_utc",
    "day_index",
    "days_since_last_notification",
    "max_template_count_last_5",
]
print([c for c in needed if c not in train.columns])
#%% Sanity baseline reward
print("train baseline:", train.select(pl.mean("session_end_completed")).item())
print("val baseline:  ", val.select(pl.mean("session_end_completed")).item())
#%% Add recency

train = add_recency(train, out_col="recency")
val   = add_recency(val,   out_col="recency")

# Keep a copy WITH history for IPS candidate evaluation later
val_for_ips = val


# now you may drop history safely (logged tables)
train = train.drop(["history"])
val   = val.drop(["history"])
#%%
train.head(10)
#%%
# One-hot encode selected_template in TRAIN
train_m = train.to_dummies(columns=["selected_template"])

val_m = val.to_dummies(columns=["selected_template"])

#%%
# ---- IMPORTANT: align val_m columns to train_m ----
missing_cols = [c for c in train_m.columns if c not in val_m.columns]
for c in missing_cols:
    val_m = val_m.with_columns(pl.lit(0).cast(pl.UInt8).alias(c))

# Drop any extra columns in val_m that are not in train_m (rare but safe)
val_m = val_m.select(train_m.columns)

assert train_m.columns == val_m.columns

#%% Build X and y
drop_cols = [
    "session_end_completed",  # label
    "eligible_templates",     # list column
]

X_train = train_m.drop(drop_cols).to_pandas()
y_train = train_m["session_end_completed"].to_pandas().astype(int)


#%% 
#%% 
# ________________RANDOM FOREST_______________ 


rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
print("RF trained.")

#%%
X_val_logged = val_m.drop(drop_cols).to_pandas()
y_val = val_m["session_end_completed"].to_pandas().astype(int)

pred_val = rf.predict_proba(X_val_logged)[:, 1]
print("Logged AUC:", roc_auc_score(y_val, pred_val))



#%% ----- Greedy policy + IPS on VAL -----

#%% create a row for each possible action (that would've been possible)
val_id = val_for_ips.with_row_index("row_id")

cand = (
    val_id
    .explode("eligible_templates")
    .rename({"eligible_templates": "candidate_template"})
    .with_columns(pl.col("candidate_template").alias("selected_template"))
)
# IMPORTANT: recompute recency for each candidate template
cand = add_recency(cand, out_col="recency")

# optional: drop history from candidates after recency is computed
cand = cand.drop(["history"])

#%% Make candidate rows look like training rows
# one-hot encode candidate template
cand_m = cand.to_dummies(columns=["selected_template"])

# align candidate columns to train_m
missing_cols = [c for c in train_m.columns if c not in cand_m.columns]
for c in missing_cols:
    cand_m = cand_m.with_columns(pl.lit(0).cast(pl.UInt8).alias(c))

cand_m = cand_m.select(train_m.columns)

#%% Predict reward for every candidate
# predict for each candidate
X_cand = cand_m.drop(drop_cols).to_pandas()
cand = cand.with_columns(pl.Series("pred", rf.predict_proba(X_cand)[:, 1]))

#%% Chose the best template per event
# choose best per row
chosen = (
    cand.sort(["row_id", "pred"], descending=[False, True])
        .group_by("row_id")
        .agg([
            pl.first("candidate_template").alias("model_choice"),
            pl.first("pred").alias("model_pred"),
        ])
)

#%% Offline evaluation: only count cases where we can observe reward
#  evaluate with IPS
val_eval = (
    val_id.join(chosen, on="row_id", how="left")
          .with_columns([
              (pl.col("model_choice") == pl.col("selected_template")).alias("match"),
              pl.col("eligible_templates").list.len().alias("pool_size"),
          ])
)

ips = val_eval.select(
    (pl.col("match").cast(pl.Float64)
     * pl.col("session_end_completed").cast(pl.Float64)
     * pl.col("pool_size").cast(pl.Float64)
    ).mean()
).item()

baseline = val_eval.select(pl.mean("session_end_completed")).item()

print("VAL baseline reward:", baseline)
print("VAL IPS reward (RF greedy):", ips)
print("Relative lift:", (ips - baseline) / baseline)


# %% Lookg for feature importance to understand session completion further. 
importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(importances.head(15))


#%% Check if templates matter at all
template_importance = importances[
    importances.index.str.startswith("selected_template_")
].sum()

print("Total template importance:", template_importance)

#%% Partial dependence: How does reward change with time since last notif
PartialDependenceDisplay.from_estimator(
    rf,
    X_train,
    ["days_since_last_notification"]
)
plt.show()

# %%

