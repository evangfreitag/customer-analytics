"""
Data Validation Script
Runs as part of CI/CD pipeline before model training.
Checks data quality, schema, and statistical properties.
"""

import pandas as pd
import numpy as np
import sys
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/orders_clean.csv"
REQUIRED_COLUMNS = ["buyer_id", "order_date", "order_value", "buyer_country"]
MIN_ROWS = 100
MIN_UNIQUE_BUYERS = 10
MAX_NULL_PCT = 0.05  # 5% max nulls per column


def validate_schema(df):
    """Check required columns exist."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print("✓ Schema validation passed")


def validate_row_count(df):
    """Check minimum row count."""
    if len(df) < MIN_ROWS:
        raise ValueError(f"Insufficient data: {len(df)} rows (minimum {MIN_ROWS})")
    print(f"✓ Row count OK: {len(df):,} rows")


def validate_nulls(df):
    """Check null rates per column."""
    for col in REQUIRED_COLUMNS:
        null_pct = df[col].isna().mean()
        if null_pct > MAX_NULL_PCT:
            raise ValueError(f"Column '{col}' has {null_pct:.1%} nulls (max {MAX_NULL_PCT:.0%})")
    print("✓ Null rate validation passed")


def validate_buyers(df):
    """Check minimum unique buyer count."""
    n_buyers = df["buyer_id"].nunique()
    if n_buyers < MIN_UNIQUE_BUYERS:
        raise ValueError(f"Insufficient unique buyers: {n_buyers} (minimum {MIN_UNIQUE_BUYERS})")
    print(f"✓ Unique buyers OK: {n_buyers:,}")


def validate_order_values(df):
    """Check order values are positive and reasonable."""
    if (df["order_value"] <= 0).any():
        raise ValueError("Negative or zero order values detected")
    if df["order_value"].max() > 100_000:
        print(f"⚠ Warning: Max order value ${df['order_value'].max():,.2f} — check for outliers")
    print(f"✓ Order value range OK: ${df['order_value'].min():.2f} – ${df['order_value'].max():,.2f}")


def validate_date_range(df):
    """Check date range is reasonable."""
    df["order_date"] = pd.to_datetime(df["order_date"])
    date_range_days = (df["order_date"].max() - df["order_date"].min()).days
    if date_range_days < 30:
        raise ValueError(f"Date range too short: {date_range_days} days")
    print(f"✓ Date range OK: {df['order_date'].min().date()} → {df['order_date'].max().date()} ({date_range_days} days)")


def validate_psi(df, reference_path="data/ebay_orders_reference.csv", threshold=0.25):
    """
    Population Stability Index — check if current data distribution
    has drifted significantly from the reference/training period.
    Skipped if no reference data exists.
    """
    if not os.path.exists(reference_path):
        print("⚠ No reference data found — skipping PSI check")
        return

    ref = pd.read_csv(reference_path)

    def psi(expected, actual, bins=10):
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        exp_counts = np.histogram(expected, bins=breakpoints)[0]
        act_counts = np.histogram(actual,   bins=breakpoints)[0]

        exp_pct = np.where(exp_counts == 0, 0.0001, exp_counts / len(expected))
        act_pct = np.where(act_counts == 0, 0.0001, act_counts / len(actual))

        return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

    psi_value = psi(ref["order_value"].dropna(), df["order_value"].dropna())

    if psi_value > threshold:
        raise ValueError(f"PSI {psi_value:.3f} exceeds threshold {threshold} — data distribution has shifted")
    elif psi_value > 0.1:
        print(f"⚠ PSI {psi_value:.3f} — moderate drift, monitor closely")
    else:
        print(f"✓ PSI OK: {psi_value:.3f}")


def main():
    print(f"\nLoading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"✗ Data file not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows\n")

    print("Running validation checks...")
    try:
        validate_schema(df)
        validate_row_count(df)
        validate_nulls(df)
        validate_buyers(df)
        validate_order_values(df)
        validate_date_range(df)
        validate_psi(df)
        print("\n✓ All data validation checks passed\n")
    except ValueError as e:
        print(f"\n✗ Data validation FAILED: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
