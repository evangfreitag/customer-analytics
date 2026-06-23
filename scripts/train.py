"""
Model Training Script
Trains BG/NBD CLV model and logs to MLflow.
Runs as part of CI/CD pipeline.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import dill as pickle
from mlflow.tracking import MlflowClient
import os
import sys
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
mlflow.set_tracking_uri("http://localhost:5000")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/orders_clean.csv"
ARTIFACTS_DIR    = "artifacts"
MODEL_NAME       = "customer-clv-bgnbd"
PENALIZER_COEF   = 0.01
PREDICTION_DAYS  = 90

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    print(f"Loaded {len(df):,} orders from {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    return df


def build_rfm_summary(df):
    snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = summary_data_from_transaction_data(
        df,
        customer_id_col="buyer_id",
        datetime_col="order_date",
        monetary_value_col="order_value",
        observation_period_end=snapshot_date,
        freq="D",
    )
    print(f"RFM summary: {len(rfm):,} customers")
    return rfm, snapshot_date


def train_model(rfm):
    bgf = BetaGeoFitter(penalizer_coef=PENALIZER_COEF)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    print("BG/NBD model fitted successfully")
    return bgf


def compute_metrics(bgf, rfm):
    """Compute model performance metrics."""
    predicted = bgf.conditional_expected_number_of_purchases_up_to_time(
        PREDICTION_DAYS, rfm["frequency"], rfm["recency"], rfm["T"]
    )
    metrics = {
        "mean_predicted_purchases_90d": float(predicted.mean()),
        "median_predicted_purchases_90d": float(predicted.median()),
        "pct_customers_predicted_active": float((predicted > 0.1).mean()),
        "n_customers": int(len(rfm)),
        "penalizer_coef": PENALIZER_COEF,
        "param_r":     float(bgf.params_["r"]),
        "param_alpha": float(bgf.params_["alpha"]),
        "param_a":     float(bgf.params_["a"]),
        "param_b":     float(bgf.params_["b"]),
    }
    return metrics


def save_artifacts(bgf, rfm):
    """Save model and RFM summary to artifacts directory."""
    model_path = os.path.join(ARTIFACTS_DIR, "bgnbd_model.pkl")
    rfm_path   = os.path.join(ARTIFACTS_DIR, "rfm_summary.csv")

    with open(model_path, "wb") as f:
        pickle.dump(bgf, f)

    rfm.to_csv(rfm_path)
    print(f"Saved model to {model_path}")
    print(f"Saved RFM summary to {rfm_path}")
    return model_path, rfm_path


def main():
    print("\n── Training BG/NBD CLV Model ──\n")

    df = load_data()
    rfm, snapshot_date = build_rfm_summary(df)
    bgf = train_model(rfm)
    metrics = compute_metrics(bgf, rfm)
    model_path, rfm_path = save_artifacts(bgf, rfm)

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="bgnbd_training") as run:
        # Log parameters
        mlflow.log_param("model_type",       "BG/NBD")
        mlflow.log_param("penalizer_coef",   PENALIZER_COEF)
        mlflow.log_param("prediction_days",  PREDICTION_DAYS)
        mlflow.log_param("snapshot_date",    str(snapshot_date.date()))
        mlflow.log_param("n_customers",      metrics["n_customers"])

        # Log metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # Log model artifact
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(rfm_path)

        # Register model in MLflow Model Registry
        client = MlflowClient()
        client.create_registered_model(MODEL_NAME) if not any(
            m.name == MODEL_NAME for m in client.search_registered_models()
        ) else None

        model_version = client.create_model_version(
            name=MODEL_NAME,
            source=f"runs:/{run.info.run_id}/bgnbd_model.pkl",
            run_id=run.info.run_id,
        )
        print(f"Model registered as: {MODEL_NAME} v{model_version.version}")

        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"Model registered as: {MODEL_NAME}")

    print("\n── Training Complete ──\n")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
