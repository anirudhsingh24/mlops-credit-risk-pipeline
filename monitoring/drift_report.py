import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("WARNING: evidently not installed. Run: pip install evidently")


REFERENCE_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "train_ref.csv")
PREDICTION_LOG  = os.path.join(os.path.dirname(__file__), "prediction_log.csv")
REPORTS_DIR     = os.path.join(os.path.dirname(__file__), "reports")
DRIFT_THRESHOLD = 0.3    # share of drifted features that triggers alert


def load_reference_data() -> pd.DataFrame:
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"Reference dataset not found at {REFERENCE_PATH}.\n"
            "Run train.py first to generate it."
        )
    df = pd.read_csv(REFERENCE_PATH)
    # drop target column if present
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    return df


def load_current_data(min_rows: int = 100) -> pd.DataFrame:
    if not os.path.exists(PREDICTION_LOG):
        raise FileNotFoundError(
            f"Prediction log not found at {PREDICTION_LOG}.\n"
            "The API must receive predictions before drift can be checked."
        )
    df = pd.read_csv(PREDICTION_LOG)
    if len(df) < min_rows:
        raise ValueError(
            f"Only {len(df)} predictions logged. Need at least {min_rows} for reliable drift detection."
        )
    # keep only feature columns (drop metadata)
    drop_cols = ["request_id", "timestamp", "probability", "decision"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols]


def run_drift_check(save_report: bool = True) -> dict:
    """Run drift detection between training reference and current prediction data."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading reference data...")
    reference = load_reference_data()

    print("Loading current prediction data...")
    current = load_current_data()

    print(f"Reference: {len(reference):,} rows | Current: {len(current):,} rows")

    if not EVIDENTLY_AVAILABLE:
        return _simple_drift_check(reference, current)

    # run Evidently report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    report.run(reference_data=reference, current_data=current)

    # save HTML report
    if save_report:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.html")
        report.save_html(report_path)
        print(f"HTML report saved → {report_path}")

    # extract key results
    results = report.as_dict()
    drift_result = results["metrics"][0]["result"]

    drifted_features = []
    all_features     = []

    if "drift_by_columns" in drift_result:
        for col, col_result in drift_result["drift_by_columns"].items():
            all_features.append(col)
            if col_result.get("drift_detected", False):
                drifted_features.append({
                    "feature":   col,
                    "p_value":   round(col_result.get("p_value", 0), 4),
                    "stat_test": col_result.get("stattest_name", "unknown")
                })

    dataset_drift   = drift_result.get("dataset_drift", False)
    share_drifted   = len(drifted_features) / max(len(all_features), 1)

    summary = {
        "timestamp":         datetime.now().isoformat(),
        "reference_rows":    len(reference),
        "current_rows":      len(current),
        "total_features":    len(all_features),
        "drifted_features":  len(drifted_features),
        "share_drifted":     round(share_drifted, 4),
        "dataset_drift":     dataset_drift,
        "alert":             dataset_drift or share_drifted >= DRIFT_THRESHOLD,
        "top_drifted":       sorted(drifted_features, key=lambda x: x["p_value"])[:5]
    }

    _print_drift_summary(summary)
    _save_drift_summary(summary)

    return summary


def _simple_drift_check(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Fallback drift check using basic statistical comparison."""
    from scipy import stats

    drifted = []
    numeric_cols = reference.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col not in current.columns:
            continue
        ref_vals  = reference[col].dropna()
        curr_vals = current[col].dropna()
        if len(ref_vals) < 10 or len(curr_vals) < 10:
            continue

        stat, p_value = stats.ks_2samp(ref_vals, curr_vals)
        if p_value < 0.05:
            drifted.append({"feature": col, "p_value": round(p_value, 4), "stat_test": "KS"})

    share_drifted = len(drifted) / max(len(numeric_cols), 1)
    summary = {
        "timestamp":        datetime.now().isoformat(),
        "reference_rows":   len(reference),
        "current_rows":     len(current),
        "total_features":   len(numeric_cols),
        "drifted_features": len(drifted),
        "share_drifted":    round(share_drifted, 4),
        "dataset_drift":    share_drifted >= DRIFT_THRESHOLD,
        "alert":            share_drifted >= DRIFT_THRESHOLD,
        "top_drifted":      sorted(drifted, key=lambda x: x["p_value"])[:5]
    }

    _print_drift_summary(summary)
    _save_drift_summary(summary)
    return summary


def _print_drift_summary(summary: dict):
    print(f"\n{'='*50}")
    print(f"  DRIFT DETECTION SUMMARY")
    print(f"{'='*50}")
    print(f"  Features checked:  {summary['total_features']}")
    print(f"  Drifted features:  {summary['drifted_features']} ({summary['share_drifted']:.1%})")
    print(f"  Dataset drift:     {'YES' if summary['dataset_drift'] else 'NO'}")

    if summary["alert"]:
        print(f"\n  ALERT: Data drift detected!")
        print(f"  Recommendation: Retrain the model on fresh data.")
        if summary["top_drifted"]:
            print(f"\n  Top drifted features:")
            for f in summary["top_drifted"]:
                print(f"    - {f['feature']} (p={f['p_value']}, test={f['stat_test']})")
    else:
        print(f"\n  OK: No significant drift detected.")
    print(f"{'='*50}\n")


def _save_drift_summary(summary: dict):
    path = os.path.join(REPORTS_DIR, "latest_drift.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Drift summary saved → {path}")


if __name__ == "__main__":
    try:
        result = run_drift_check()
        if result["alert"]:
            sys.exit(2)    # exit code 2 = drift alert (use in CI to notify)
    except FileNotFoundError as e:
        print(f"Skipping drift check: {e}")