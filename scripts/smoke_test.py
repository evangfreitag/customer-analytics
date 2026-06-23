"""
Smoke Test Script
Runs after production deployment to verify the model is serving correctly.
Loads the production model from MLflow and runs a basic prediction check.
"""

import numpy as np
import pickle
import sys
import os
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME      = "customer-clv-bgnbd"
PREDICTION_DAYS = 90


def load_production_model():
    """Load the current production model from MLflow registry."""
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if not versions:
        raise ValueError(f"No production model found for '{MODEL_NAME}'")

    version = versions[0]
    print(f"Loading {MODEL_NAME} v{version.version} from Production")

    model_uri = f"models:/{MODEL_NAME}/Production"
    # Download artifact
    local_path = mlflow.artifacts.download_artifacts(
        run_id=version.run_id,
        artifact_path="bgnbd_model.pkl"
    )

    with open(local_path, "rb") as f:
        bgf = pickle.load(f)

    return bgf, version.version


def run_smoke_test(bgf):
    """
    Basic sanity checks on the production model.
    Uses synthetic inputs to verify the model produces valid predictions.
    """
    # Synthetic test cases
    test_cases = [
        {"frequency": 0,  "recency": 0,   "T": 30,  "desc": "new one-time buyer"},
        {"frequency": 5,  "recency": 20,  "T": 30,  "desc": "active repeat buyer"},
        {"frequency": 10, "recency": 180, "T": 365, "desc": "lapsed high-frequency buyer"},
        {"frequency": 25, "recency": 510,   "T": 512, "desc": "VIP buyer (recent)"},
    ]

    print("\nRunning prediction smoke tests:")
    all_passed = True

    for tc in test_cases:
        try:
            pred = bgf.conditional_expected_number_of_purchases_up_to_time(
                PREDICTION_DAYS,
                tc["frequency"],
                tc["recency"],
                tc["T"],
            )
            # Basic sanity: prediction must be non-negative and finite
            assert pred >= 0,        f"Negative prediction: {pred}"
            assert np.isfinite(pred), f"Non-finite prediction: {pred}"
            print(f"  ✓ {tc['desc']}: predicted {pred:.4f} purchases in {PREDICTION_DAYS} days")

        except AssertionError as e:
            print(f"  ✗ {tc['desc']}: FAILED — {e}")
            all_passed = False

    return all_passed


def main():
    print("\n── Production Smoke Test ──\n")

    try:
        bgf, version = load_production_model()
        passed = run_smoke_test(bgf)

        if passed:
            print(f"\n✓ Smoke test passed — {MODEL_NAME} v{version} is serving correctly\n")
        else:
            print(f"\n✗ Smoke test FAILED — {MODEL_NAME} v{version} may have issues\n")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Smoke test error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
