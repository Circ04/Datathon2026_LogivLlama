#%% 
import polars as pl
#%%
train = pl.read_parquet("data_splits/train_sample.parquet")
val = pl.read_parquet("data_splits/val_sample.parquet")

#%%
print(train.shape)
print(train.columns)
train.select(pl.mean("session_end_completed"))
# %%
full_train = pl.read_parquet("data_splits/train.parquet")
full_train.select(pl.mean("session_end_completed"))

#%% 
full_val = pl.read_parquet("data_splits/val.parquet")
# %%
train_rsample = full_train.sample(n=100_000, seed=42)
val_rsample= full_val.sample(n=100000, seed=42 )
# %%
train_rsample.write_parquet("train_rsample_100k.parquet")
val_rsample.write_parquet("val_rsample_100k.parquet")
#%
