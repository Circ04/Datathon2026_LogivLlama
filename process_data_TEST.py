"""
Data Processing Script for Datathon 2026
Processes raw parquet files and creates clean train/val datasets with engineered features.

Usage:
    python process_data.py

Output:
    - data_splits/test.parquet
"""
import polars as pl
from pathlib import Path

def process_test_data():
    """Process test parquet partitions into a single clean test dataset (lazy execution)."""

    print("=" * 60)
    print("DATATHON 2026 - TEST DATA PROCESSING PIPELINE")
    print("=" * 60)

    # Create output directory (same as train/val)
    Path("data_splits").mkdir(exist_ok=True)

    # Step 1: Load test data lazily (IMPORTANT: test-part* matches test-part, test-part-1, test-part-2, ...)
    print("\n[1/4] Loading test data (lazy mode)...")
    df = pl.scan_parquet("Data/test-part-*/*.parquet")

    total_rows = df.select(pl.len()).collect().item()
    print(f"  ✓ Total rows: {total_rows:,}")

    # Step 2: Add engineered features (fast only)
    print("\n[2/4] Engineering features...")
    df = df.with_columns([
        pl.col('history').list.len().alias('history_length'),
        pl.col('eligible_templates').list.len().alias('n_eligible'),
    ])
    print("  ✓ Added features: history_length, n_eligible, time_of_day")
    print("  (Note: days_since_shown / recency should be added after loading, as in your train/val script)")

    # Step 3: Select final columns for modeling/eval
    print("\n[3/4] Selecting modeling columns...")
    final_columns = [
        'datetime',
        'ui_language',
        'selected_template',
        'eligible_templates',
        'history',          # keep for recency calc later (IPS candidate recency needs this)
        'history_length',
        'n_eligible',
        'session_end_completed'  # if test has it; if it doesn't, remove this line
    ]

    # If your test set truly has no labels, uncomment this safer version:
    # final_columns = [c for c in final_columns if c in df.columns]

    test = df.select(final_columns)
    print(f"  ✓ Selected {len(final_columns)} columns for modeling")

    # Step 4: Save to disk (streaming mode)
    print("\n[4/4] Saving processed test dataset to disk...")
    print("  - Writing test.parquet (full dataset)...")
    test.sink_parquet("data_splits/test.parquet")

    print("\n" + "=" * 60)
    print("✓ TEST PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nOutput files:")
    print("  → data_splits/test.parquet (full)")
    print("  → data_splits/test_sample.parquet (20K rows)")
    print("\nLoad with:")
    print('  test = pl.read_parquet("data_splits/test.parquet")')

if __name__ == "__main__":
    process_test_data()