import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(__file__))
from ingest import load_and_prepare
from features import build_features


MODEL_NAME      = "credit-risk-model"
AUC_THRESHOLD   = 0.72     # minimum acceptable AUC — hard floor
IMPROVEMENT_MIN = 0.003    # new model must beat prod by at least 0.3%
DATA_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "credit.csv")
ARTIFACTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def load_production_model(client: MlflowClient):
    """Load the current Production model from MLflow registry."""
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            print("No Production model found — this will be the first deployment.")
            return None, None
        prod_version = versions[0]
        model_uri    = f"models:/{MODEL_NAME}/Production"
        model        = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded Production model: v{prod_version.version}")
        return model, prod_version
    except Exception as e:
        print(f"Could not load Production model: {e}")
        return None, None


def load_candidate_model(client: MlflowClient):
    """Load the latest undeployed (stage=None) model from the registry."""
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if not versions:
        raise ValueError(
            "No candidate model found in registry.\n"
            "Run: python src/train.py  first."
        )
    # take the highest version number
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
    model     = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded candidate model: v{latest.version} (run_id: {latest.run_id[:8]}...)")
    return model, latest


def compute_metrics(model, X_test, y_test) -> dict:
    """Compute full evaluation metrics."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc":                round(roc_auc_score(y_test, probs), 4),
        "f1":                 round(f1_score(y_test, preds), 4),
        "precision":          round(precision_score(y_test, preds), 4),
        "recall":             round(recall_score(y_test, preds), 4),
        "true_positives":     int(tp),
        "false_positives":    int(fp),
        "true_negatives":     int(tn),
        "false_negatives":    int(fn),
        "false_negative_rate": round(fn / (fn + tp + 1e-6), 4),  # missed defaulters
        "false_positive_rate": round(fp / (fp + tn + 1e-6), 4),  # false alarms
    }


def print_metrics(label: str, metrics: dict):
    print(f"\n{'─'*45}")
    print(f"  {label}")
    print(f"{'─'*45}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print(f"{'─'*45}")


def promote_model(client: MlflowClient, candidate_version, metrics: dict):
    """Archive current Production, promote candidate, tag with metrics."""
    # archive current production models
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived"
        )
        print(f"Archived previous Production v{v.version}")

    # promote candidate to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=candidate_version.version,
        stage="Production"
    )

    # tag with eval metrics for traceability
    for metric_name, value in metrics.items():
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=candidate_version.version,
            key=f"eval_{metric_name}",
            value=str(value)
        )

    print(f"Promoted v{candidate_version.version} to Production.")

    # save promotion record
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    record = {
        "promoted_version": candidate_version.version,
        "run_id":           candidate_version.run_id,
        "metrics":          metrics
    }
    with open(os.path.join(ARTIFACTS_DIR, "last_promotion.json"), "w") as f:
        json.dump(record, f, indent=2)
    print("Promotion record saved → artifacts/last_promotion.json")


def evaluate():
    client = MlflowClient()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # load test data — use a fixed random_state so train and evaluate
    # always use the same split
    df = load_and_prepare(DATA_PATH)
    X, y = build_features(df)

    # replicate the same 80/20 split used in train.py
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # load models
    prod_model, prod_version       = load_production_model(client)
    cand_model, cand_version       = load_candidate_model(client)

    # score candidate
    cand_metrics = compute_metrics(cand_model, X_test, y_test)
    print_metrics("CANDIDATE MODEL", cand_metrics)

    # ── gate 1: absolute AUC floor ──────────────────────
    if cand_metrics["auc"] < AUC_THRESHOLD:
        print(f"\nFAIL ✗  Candidate AUC {cand_metrics['auc']} is below "
              f"minimum threshold {AUC_THRESHOLD}.")
        print("Pipeline blocked. Fix the model and retrain.")
        sys.exit(1)

    # ── gate 2: improvement over production ─────────────
    if prod_model is not None:
        prod_metrics = compute_metrics(prod_model, X_test, y_test)
        print_metrics("PRODUCTION MODEL", prod_metrics)

        improvement = cand_metrics["auc"] - prod_metrics["auc"]
        print(f"\n  AUC improvement: {improvement:+.4f}  (required: ≥ +{IMPROVEMENT_MIN})")

        if improvement < IMPROVEMENT_MIN:
            print(f"\nFAIL ✗  Candidate does not improve sufficiently over Production.")
            print(f"  Current Production AUC: {prod_metrics['auc']}")
            print(f"  Candidate AUC:          {cand_metrics['auc']}")
            print("Pipeline blocked. No deployment.")
            sys.exit(1)

    # ── all gates passed ─────────────────────────────────
    print("\nPASS ✓  All quality gates passed.")
    promote_model(client, cand_version, cand_metrics)

    # save candidate metrics for CI logs
    with open(os.path.join(ARTIFACTS_DIR, "candidate_metrics.json"), "w") as f:
        json.dump(cand_metrics, f, indent=2)

    return cand_metrics


if __name__ == "__main__":
    metrics = evaluate()
    print("\nEvaluation complete.")
    print(json.dumps(metrics, indent=2))