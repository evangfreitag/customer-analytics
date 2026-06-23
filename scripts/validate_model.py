"""
Model Validation Script
Validates trained BG/NBD model before promotion to staging.
Checks performance metrics, parameter reasonableness, and PSI.
Runs as part of CI/CD pipeline after training.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os

# ── Validation Thresholds ────────────────────────────────────────────────────
THRESHOLDS = {
    "min_customers":              10,      # minimum customers in RFM summary
    "max_psi":                    0.25,    # maximum PSI before flagging drift
    "min_pct_active":             0.01,    # at least 1% predicted active
    "max_pct_active":             0.99,    # sanity check — not everyone active
    "min_param_r":                0.001,   # BG/NBD parameter sanity checks
    "max_param_r":                10.0,
    "min_param_alpha":            0.001,
    "max_param_alpha":            1000.0,
}

ARTIFACTS_DIR = "artifacts"
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "bgnbd_model.pkl")
RFM_PATH      = os.path.join(ARTIFACTS_DIR, "rfm_summary.csv")
PREDICTION_DAYS = 90


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(RFM_PATH):
        raise FileNotFoundError(f"RFM summary not found: {RFM_PATH}")

    with open(MODEL_PATH, "rb") as f:
        bgf = pickle.load(f)

    rfm = pd.read_csv(RFM_PATH, index_col=0)
    print(f"✓ Artifacts loaded: {len(rfm):,} customers")
    return bgf, rfm


def validate_customer_count(rfm):
    n = len(rfm)
    if n < THRESHOLDS["min_customers"]:
        raise ValueError(f"Too few customers: {n} (minimum {THRESHOLDS['min_customers']})")
    print(f"✓ Customer count OK: {n:,}")


def validate_model_parameters(bgf):
    """Check BG/NBD parameters are within reasonable bounds."""
    params = bgf.params_
    checks = [
        ("r",     THRESHOLDS["min_param_r"],     THRESHOLDS["max_param_r"]),
        ("alpha", THRESHOLDS["min_param_alpha"],  THRESHOLDS["max_param_alpha"]),
    ]
    for param, lo, hi in checks:
        val = params[param]
        if not (lo <= val <= hi):
            raise ValueError(f"Parameter '{param}' = {val:.4f} outside range [{lo}, {hi}]")
    print(f"✓ Model parameters OK: r={params['r']:.4f}, alpha={params['alpha']:.4f}, "
          f"a={params['a']:.4f}, b={params['b']:.4f}")


def validate_predictions(bgf, rfm):
    """Check predicted purchase distribution is sensible."""
    predicted = bgf.conditional_expected_number_of_purchases_up_to_time(
        PREDICTION_DAYS, rfm["frequency"], rfm["recency"], rfm["T"]
    )

    pct_active = (predicted > 0.1).mean()

    if pct_active < THRESHOLDS["min_pct_active"]:
        raise ValueError(f"Only {pct_active:.1%} customers predicted active — model may be degenerate")
    if pct_active > THRESHOLDS["max_pct_active"]:
        raise ValueError(f"{pct_active:.1%} customers predicted active — model may be overfit")

    print(f"✓ Predictions OK: {pct_active:.1%} customers predicted to purchase in next {PREDICTION_DAYS} days")
    print(f"  Mean predicted purchases: {predicted.mean():.4f}")
    print(f"  Median predicted purchases: {predicted.median():.4f}")


def validate_psi(rfm, reference_path="data/ebay_orders_reference.csv"):
    """PSI on monetary value distribution vs reference period."""
    if not os.path.exists(reference_path):
        print("⚠ No reference data — skipping PSI validation")
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

    psi_val = psi(ref["monetary"].dropna(), rfm["monetary"].dropna())

    if psi_val > THRESHOLDS["max_psi"]:
        raise ValueError(f"PSI {psi_val:.3f} exceeds threshold {THRESHOLDS['max_psi']} — population has shifted")
    elif psi_val > 0.1:
        print(f"⚠ PSI {psi_val:.3f} — moderate drift, monitor")
    else:
        print(f"✓ PSI OK: {psi_val:.3f}")


def main():
    print("\n── Model Validation ──\n")

    try:
        bgf, rfm = load_artifacts()
        validate_customer_count(rfm)
        validate_model_parameters(bgf)
        validate_predictions(bgf, rfm)
        validate_psi(rfm)
        print("\n✓ All model validation checks passed — ready for staging\n")

    except (ValueError, FileNotFoundError) as e:
        print(f"\n✗ Model validation FAILED: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
