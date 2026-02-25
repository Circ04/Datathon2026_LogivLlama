#%%
## _________________Final Evaluation RF on Test set_____________________________
## Do logged AUC on full test set. Since IPS uses exploding observations, 
# evaluation isn't possible on full test, so take a random 500k subset
import polars as pl
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from collections import Counter

#%% 
TRAIN_PATH = "train_preprocessed_100k.parquet"
VAL_PATH   = "val_preprocessed_100k.parquet"
TEST_FULL_PARQUET = "test_rsample_7mio.parquet"

#%%

train = pl.read_parquet(TRAIN_PATH)
val   = pl.read_parquet(VAL_PATH)
test  = pl.read_parquet(TEST_FULL_PARQUET)

# %%
TEST_IPS_N = 500_000             # IPS subset size
TEST_AUC_BATCH = 250_000         # batch size for full-test AUC (tune for RAM/speed)

SENTINEL_RECENCY = 100.0         # keep consistent with your modeling pipeline



#%% ----------------- Make TEST match the preprocessing of TRAIN/VAL -----------------
#%%
# 1) Merge train + val (they already have identical schema from preprocessing)
train_full = pl.concat([train, val], how="vertical_relaxed")
print("train_full:", train_full.shape, "test:", test.shape)

#%%
# 2) Ensure hour_utc exists (your preprocess_A.py created it)
def add_hour_utc(df: pl.DataFrame) -> pl.DataFrame:
    if "hour_utc" in df.columns:
        return df
    return df.with_columns(
        ((pl.col("datetime") % 1) * 24)
          .floor()
          .cast(pl.Int8)
          .alias("hour_utc")
    )

#%%
# 3) Ensure day_index exists
def add_day_index(df: pl.DataFrame) -> pl.DataFrame:
    if "day_index" in df.columns:
        return df
    return df.with_columns(
        (pl.col("datetime").floor().cast(pl.Int32) + 1).alias("day_index")
    )

#%%
# 4) Ensure days_since_last_notification exists (based on ANY template)
def add_days_since_last_notification(df: pl.DataFrame) -> pl.DataFrame:
    if "days_since_last_notification" in df.columns:
        return df
    return df.with_columns(
        pl.col("history").map_elements(
            lambda hist: float(SENTINEL_RECENCY) if (hist is None or len(hist) == 0)
            else float(min(item["n_days"] for item in hist)),
            return_dtype=pl.Float64
        ).alias("days_since_last_notification")
    )

#%%
# 5) Ensure max_template_count_last_5 exists
def add_max_template_count_last_5(df: pl.DataFrame) -> pl.DataFrame:
    if "max_template_count_last_5" in df.columns:
        return df
    return df.with_columns(
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
# 6) Ensure ui_language is one-hot encoded AND aligned to train_full
#    train_full already has ui_language_* columns (because you saved preprocessed train/val)
lang_cols = [c for c in train_full.columns if c.startswith("ui_language_")]

def ensure_ui_language_dummies(df: pl.DataFrame, ref_lang_cols: list[str]) -> pl.DataFrame:
    # If already dummy-encoded (no ui_language column), just align columns
    if "ui_language" not in df.columns:
        # add missing language dummy cols
        missing = [c for c in ref_lang_cols if c not in df.columns]
        for c in missing:
            df = df.with_columns(pl.lit(0).cast(pl.UInt8).alias(c))
        return df

    # If ui_language exists, create dummies then align to ref
    df = df.to_dummies(columns=["ui_language"])
    missing = [c for c in ref_lang_cols if c not in df.columns]
    for c in missing:
        df = df.with_columns(pl.lit(0).cast(pl.UInt8).alias(c))

    # drop any extra ui_language_* not seen in train_full (rare but safe)
    extra = [c for c in df.columns if c.startswith("ui_language_") and c not in ref_lang_cols]
    if extra:
        df = df.drop(extra)

    return df

#%%
# Apply preprocessing to TEST only (train_full already has these from preprocess_A.py)
test = add_hour_utc(test)
test = add_day_index(test)
test = add_days_since_last_notification(test)
test = add_max_template_count_last_5(test)
test = ensure_ui_language_dummies(test, lang_cols)

#%%
#Optional sanity: confirm test now contains everything your model expects (minus recency, which comes next)
needed_now = [
    "session_end_completed",
    "selected_template",
    "eligible_templates",
    "history",
    "hour_utc",
    "day_index",
    "days_since_last_notification",
    "max_template_count_last_5",
]
missing_now = [c for c in needed_now if c not in test.columns]
print("Missing in test after preprocess:", missing_now)
# %%








#%% ----------------- NEXT STEP: mimic 3_Modeling.py from here -----------------

#%%
# 1) Recency helper (same as 3_Modeling.py, sentinel = 100.0 in your setup)
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

#%%
# 2) Add recency to TRAIN_FULL and TEST (both still have history)
train_full = add_recency(train_full, out_col="recency")
test       = add_recency(test,       out_col="recency")

# Keep a copy WITH history for IPS later
test_for_ips = test

#%%
# 3) Drop history for logged model matrices
train_logged = train_full.drop(["history"])
test_logged  = test.drop(["history"])

#%%
# 4) One-hot encode selected_template and ALIGN test columns to train columns
train_m = train_logged.to_dummies(columns=["selected_template"])
test_m  = test_logged.to_dummies(columns=["selected_template"])

missing_cols = [c for c in train_m.columns if c not in test_m.columns]
for c in missing_cols:
    test_m = test_m.with_columns(pl.lit(0).cast(pl.UInt8).alias(c))

# drop any extras and order columns exactly like train_m
test_m = test_m.select(train_m.columns)
#%%
if "day_index" in train_m.columns:
    train_m = train_m.drop(["day_index"])
if "day_index" in test_m.columns:
    test_m = test_m.drop(["day_index"])
#%%
# 5) Train RF on train+val
drop_cols = ["session_end_completed", "eligible_templates"]

X_train = train_m.drop(drop_cols).to_pandas()
y_train = train_m["session_end_completed"].to_pandas().astype(int)

rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print("RF trained on train+val.")

#%%
# 6) Logged AUC on TEST (your test here is 7M rows — OK)
X_test_logged = test_m.drop(drop_cols).to_pandas()
y_test = test_m["session_end_completed"].to_pandas().astype(int)

pred_test = rf.predict_proba(X_test_logged)[:, 1]
print("TEST Logged AUC:", roc_auc_score(y_test, pred_test))

#%%
# 7) IPS on a random 500k subset of TEST (to avoid exploding the full test)
TEST_IPS_N = 1000_000
test_ips = test_for_ips.sample(n=TEST_IPS_N, shuffle=True, seed=42)

test_id = test_ips.with_row_index("row_id")

cand = (
    test_id
    .explode("eligible_templates")
    .rename({"eligible_templates": "candidate_template"})
    .with_columns(pl.col("candidate_template").alias("selected_template"))
)

#%%
# IMPORTANT: recompute recency for candidate template (needs history)
cand = add_recency(cand, out_col="recency")

# drop history now that recency is computed
cand = cand.drop(["history"])

# one-hot encode candidates and align to train_m
cand_m = cand.to_dummies(columns=["selected_template"])

missing_cols = [c for c in train_m.columns if c not in cand_m.columns]
for c in missing_cols:
    cand_m = cand_m.with_columns(pl.lit(0).cast(pl.UInt8).alias(c))

cand_m = cand_m.select(train_m.columns)

# predict on candidates
X_cand = cand_m.drop(drop_cols).to_pandas()
cand = cand.with_columns(pl.Series("pred", rf.predict_proba(X_cand)[:, 1]))

# choose best candidate per event
chosen = (
    cand.sort(["row_id", "pred"], descending=[False, True])
        .group_by("row_id")
        .agg([
            pl.first("candidate_template").alias("model_choice"),
            pl.first("pred").alias("model_pred"),
        ])
)

#%%
# IPS estimate
test_eval = (
    test_id.join(chosen, on="row_id", how="left")
           .with_columns([
               (pl.col("model_choice") == pl.col("selected_template")).alias("match"),
               pl.col("eligible_templates").list.len().alias("pool_size"),
           ])
)

baseline = test_eval.select(pl.mean("session_end_completed")).item()
ips = test_eval.select(
    (pl.col("match").cast(pl.Float64)
     * pl.col("session_end_completed").cast(pl.Float64)
     * pl.col("pool_size").cast(pl.Float64)
    ).mean()
).item()

print("TEST baseline reward (IPS sample):", baseline)
print("TEST IPS reward (RF greedy, IPS sample):", ips)
print("Relative lift:", (ips - baseline) / baseline)
# %%