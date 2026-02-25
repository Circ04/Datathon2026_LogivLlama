#%% 
import polars as pl

# %%
full_train = pl.read_parquet("data_splits/train.parquet")
full_train.select(pl.mean("session_end_completed"))

full_train = full_train.with_columns(
    ((pl.col("datetime") - pl.col("datetime").floor()) * 24).alias("hour_utc")
)


#%% 
full_val = pl.read_parquet("data_splits/val.parquet")
full_val = full_val.with_columns(
    ((pl.col("datetime") - pl.col("datetime").floor()) * 24).alias("hour_utc")
)
# %%
train_rsample = full_train.sample(n=100_000, seed=42)
val_rsample= full_val.sample(n=100000, seed=42 )
# %%
train_rsample.write_parquet("train_rsample_100k.parquet")
val_rsample.write_parquet("val_rsample_100k.parquet")
#%


#%%
#  If you've written a single giant parquet:
TEST_FULL_PARQUET = "test.parquet"
test  = pl.read_parquet(TEST_FULL_PARQUET)
#%%
test_rsample= test.sample(n=7000000, seed=42 )
#%%
test_rsample.write_parquet("test_rsample_7mio.parquet")
# %%
