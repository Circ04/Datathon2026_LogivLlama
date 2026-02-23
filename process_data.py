"""
Data Processing Script for Datathon 2026
Processes raw parquet files and creates clean train/val datasets with engineered features.

Usage:
    python process_data.py

Output:
    - data_splits/train.parquet
    - data_splits/val.parquet
"""

import polars as pl
from pathlib import Path

def get_days_since_shown(row):
    """
    Extract days since the selected template was last shown.

    Args:
        row: Dict with 'selected_template' and 'history' keys

    Returns:
        float: Minimum n_days for matching template, or 999.0 if never shown
    """
    template = row['selected_template']
    history = row['history']

    # Find all history entries where template matches
    matching_days = [float(h['n_days']) for h in history if h['template'] == template]

    # Return minimum (most recent) or 999.0 if never shown
    return float(min(matching_days)) if matching_days else 999.0


def process_data():
    """Main processing function - runs everything with lazy execution."""

    print("=" * 60)
    print("DATATHON 2026 - DATA PROCESSING PIPELINE")
    print("=" * 60)

    # Create output directory
    Path("data_splits").mkdir(exist_ok=True)

    # Step 1: Load data lazily
    print("\n[1/5] Loading training data (lazy mode)...")
    df = pl.scan_parquet("Data/train-part/*.parquet")

    # Get basic stats without loading all data
    total_rows = df.select(pl.len()).collect().item()
    print(f"  ✓ Total rows: {total_rows:,}")

    # Step 2: Add engineered features
    print("\n[2/5] Engineering features...")
    print("  (Using native Polars expressions for speed...)")

    df = df.with_columns([
        # Additional features (fast)
        pl.col('history').list.len().alias('history_length'),
        pl.col('eligible_templates').list.len().alias('n_eligible'),
        pl.col('datetime').alias('time_of_day'),
    ])

    # For days_since_shown, we'll compute it after collecting
    # This avoids the slow map_elements on the full dataset
    print("  ✓ Added features: history_length, n_eligible, time_of_day")
    print("  (Note: days_since_shown will be added after loading - see note below)")

    # Step 3: Create train/val splits (by time)
    print("\n[3/5] Creating time-based train/val splits...")
    print("  - Train: datetime <= 11 days")
    print("  - Val: datetime > 11 days")

    train = df.filter(pl.col("datetime") <= 11)
    val = df.filter(pl.col("datetime") > 11)

    # Check engagement rates
    train_engagement = train.select(pl.col("session_end_completed").mean()).collect().item()
    val_engagement = val.select(pl.col("session_end_completed").mean()).collect().item()

    print(f"  ✓ Train engagement rate: {train_engagement:.4%}")
    print(f"  ✓ Val engagement rate: {val_engagement:.4%}")

    # Step 4: Select final columns for modeling
    print("\n[4/5] Selecting modeling columns...")
    final_columns = [
        'datetime',
        'ui_language',
        'selected_template',
        'eligible_templates',
        'history',  # Keep history for post-processing
        'history_length',
        'n_eligible',
        'time_of_day',
        'session_end_completed'  # Target variable
    ]

    train = train.select(final_columns)
    val = val.select(final_columns)
    print(f"  ✓ Selected {len(final_columns)} columns for modeling")

    # Step 5: Save to disk (streaming mode - no RAM issues)
    print("\n[5/5] Saving processed datasets to disk...")
    print("  - Writing train.parquet (full dataset)...")
    train.sink_parquet("data_splits/train.parquet")

    print("  - Writing val.parquet (full dataset)...")
    val.sink_parquet("data_splits/val.parquet")

    # Also create smaller samples for exploration
    print("\n[BONUS] Creating sample datasets for fast exploration...")
    print("  - Creating train_sample.parquet (100K rows)...")
    train.head(100_000).sink_parquet("data_splits/train_sample.parquet")

    print("  - Creating val_sample.parquet (20K rows)...")
    val.head(20_000).sink_parquet("data_splits/val_sample.parquet")

    print("\n" + "=" * 60)
    print("✓ PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nOutput files:")
    print("  → data_splits/train.parquet (full)")
    print("  → data_splits/val.parquet (full)")
    print("  → data_splits/train_sample.parquet (100K rows - for exploration)")
    print("  → data_splits/val_sample.parquet (20K rows - for exploration)")
    print("\nFor FAST exploration, use the samples:")
    print('  train = pl.read_parquet("data_splits/train_sample.parquet")')
    print("\nFor FINAL training, use the full datasets:")
    print('  train = pl.read_parquet("data_splits/train.parquet")')
    print("\nTo add days_since_shown feature:")
    print("""
import polars as pl

# Load data
train = pl.read_parquet("data_splits/train.parquet")

# Add days_since_shown feature (on smaller dataset, much faster)
def get_days_since_shown(row):
    template = row['selected_template']
    history = row['history']
    matching_days = [float(h['n_days']) for h in history if h['template'] == template]
    return float(min(matching_days)) if matching_days else 999.0

train = train.with_columns(
    pl.struct(['selected_template', 'history'])
      .map_elements(get_days_since_shown, return_dtype=pl.Float64)
      .alias('days_since_shown')
)

# Now ready for modeling!
    """)


if __name__ == "__main__":
    process_data()
