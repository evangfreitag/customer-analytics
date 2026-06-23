"""
Model Promotion Script
Promotes a registered MLflow model between stages:
None → Staging → Production

Runs as part of CI/CD pipeline.
Staging promotion is automatic after validation.
Production promotion requires manual approval (GitHub environment gate).
"""

import argparse
import sys
import mlflow
from mlflow.tracking import MlflowClient


def promote_model(model_name: str, to_stage: str, version: int = None):
    """
    Promote the latest model version to the target stage.

    Args:
        model_name: Name of the registered model in MLflow
        to_stage:   Target stage — 'Staging' or 'Production'
        version:    Specific version to promote (optional, defaults to latest)
    """
    client = MlflowClient()

    # Map to_stage to from_stage
    from_stage_map = {
        "Staging":    "None",
        "Production": "Staging",
    }

    if to_stage not in from_stage_map:
        raise ValueError(f"Invalid stage: {to_stage}. Must be 'Staging' or 'Production'")

    from_stage = from_stage_map[to_stage]

    # Get version to promote
    if version is None:
        versions = client.get_latest_versions(model_name, stages=[from_stage])
        if not versions:
            # For Staging, also check 'None' stage
            versions = client.get_latest_versions(model_name, stages=["None"])
        if not versions:
            raise ValueError(f"No model versions found in stage '{from_stage}' for model '{model_name}'")
        version = versions[0].version

    print(f"Promoting {model_name} v{version}: {from_stage} → {to_stage}")

    # Archive existing versions in target stage
    existing = client.get_latest_versions(model_name, stages=[to_stage])
    for v in existing:
        print(f"  Archiving existing {to_stage} version {v.version}")
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Archived",
        )

    # Promote new version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=to_stage,
        archive_existing_versions=True,
    )

    print(f"✓ Successfully promoted {model_name} v{version} to {to_stage}")

    # Log promotion event
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key=f"promoted_to_{to_stage.lower()}",
        value="true",
    )


def main():
    parser = argparse.ArgumentParser(description="Promote MLflow model between stages")
    parser.add_argument("--model-name", required=True, help="Registered model name")
    parser.add_argument("--to-stage",   required=True, choices=["Staging", "Production"])
    parser.add_argument("--version",    type=int,      help="Specific version (optional)")
    args = parser.parse_args()

    try:
        promote_model(args.model_name, args.to_stage, args.version)
    except Exception as e:
        print(f"✗ Promotion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
