import polars as pl
import numpy as np

df = pl.read_parquet("Data/training/part-00000-9b4bba6b-feac-44b1-a155-17c796835cca-c000.snappy.parquet")
print(df.head())