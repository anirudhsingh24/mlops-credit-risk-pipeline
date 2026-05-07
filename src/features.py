import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS       = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_COLS        = ["PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",  "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6"]

TARGET = "default.payment.next.month"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


# ─────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- basic aggregates ---
    df["avg_pay_delay"]   = df[PAY_STATUS_COLS].mean(axis=1)
    df["max_pay_delay"]   = df[PAY_STATUS_COLS].max(axis=1)
    df["months_delayed"]  = (df[PAY_STATUS_COLS] > 0).sum(axis=1)

    # --- repayment ratio: total paid vs total billed ---
    total_billed = df[BILL_COLS].sum(axis=1)
    total_paid   = df[PAY_COLS].sum(axis=1)
    df["repayment_ratio"] = np.where(
        total_billed > 0,
        total_paid / (total_billed + 1e-6),
        0.0
    )

    # --- credit utilisation: latest bill vs credit limit ---
    df["credit_utilisation"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1e-6)
    df["credit_utilisation"] = df["credit_utilisation"].clip(0, 2)

    # --- trend features ---
    df["bill_trend"]    = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["payment_trend"] = df["PAY_AMT1"]  - df["PAY_AMT6"]
    df["bill_velocity"] = (df["BILL_AMT1"] - df["BILL_AMT2"]) / (df["BILL_AMT2"] + 1e-6)

    # --- volatility features ---
    df["bill_std"]    = df[BILL_COLS].std(axis=1)
    df["payment_std"] = df[PAY_COLS].std(axis=1)
    df["bill_cv"]     = df["bill_std"]    / (df[BILL_COLS].mean(axis=1) + 1e-6)
    df["payment_cv"]  = df["payment_std"] / (df[PAY_COLS].mean(axis=1)  + 1e-6)

    # --- interaction features ---
    df["age_limit_interaction"]  = df["AGE"] * df["LIMIT_BAL"]
    df["util_delay_interaction"] = df["credit_utilisation"] * df["avg_pay_delay"]
    df["payment_gap_last_month"] = df["BILL_AMT1"] - df["PAY_AMT1"]
    df["payment_gap_ratio"]      = df["payment_gap_last_month"] / (df["BILL_AMT1"] + 1e-6)

    # --- consecutive delay streak ---
    df["consecutive_delay_streak"] = df.apply(_consecutive_delays, axis=1)

    # --- recency-weighted payment delay ---
    weights = [0.35, 0.25, 0.18, 0.12, 0.06, 0.04]
    df["weighted_pay_delay"] = sum(
        w * df[col] for w, col in zip(weights, PAY_STATUS_COLS)
    )

    # --- monthly pay ratios ---
    # captures repayment discipline for each individual month
    # e.g. pay_ratio_1=0.02 means customer only paid 2% of their bill last month
    for i in range(1, 7):
        df[f"pay_ratio_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"] + 1e-6)
        df[f"pay_ratio_{i}"] = df[f"pay_ratio_{i}"].clip(0, 2)

    # --- deterioration: is behaviour improving or worsening? ---
    # compares recent 3 months vs older 3 months of payment delays
    # positive value = getting worse recently = strong default risk signal
    df["recent_pay_avg"] = df[["PAY_0", "PAY_2", "PAY_3"]].mean(axis=1)
    df["older_pay_avg"]  = df[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    df["deterioration"]  = df["recent_pay_avg"] - df["older_pay_avg"]

    # --- months underpaid ---
    # minimum payment is typically 1-2% of bill
    # customer paying less than minimum = serious risk signal
    for i in range(1, 7):
        df[f"underpaid_{i}"] = (
            df[f"PAY_AMT{i}"] < df[f"BILL_AMT{i}"] * 0.02
        ).astype(int)
    df["months_underpaid"] = df[[f"underpaid_{i}" for i in range(1, 7)]].sum(axis=1)

    # --- utilisation trend ---
    # is the customer slowly maxing out their card over time?
    # positive = utilisation growing = classic pre-default pattern
    df["util_recent"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1e-6)
    df["util_older"]  = df["BILL_AMT6"] / (df["LIMIT_BAL"] + 1e-6)
    df["util_trend"]  = df["util_recent"] - df["util_older"]

    # --- squared utilisation ---
    # captures non-linear risk: going from 90% to 100% utilisation
    # is far more dangerous than going from 50% to 60%
    df["credit_utilisation_sq"] = df["credit_utilisation"] ** 2

    return df


def _consecutive_delays(row) -> int:
    """Count consecutive months of late payment starting from most recent."""
    streak = 0
    for col in PAY_STATUS_COLS:
        if row[col] > 0:
            streak += 1
        else:
            break
    return streak


# ─────────────────────────────────────────────
#  Feature column lists
# ─────────────────────────────────────────────

ENGINEERED_FEATURES = [
    # basic aggregates
    "avg_pay_delay", "max_pay_delay", "months_delayed",
    # ratios
    "repayment_ratio", "credit_utilisation",
    # trends
    "bill_trend", "payment_trend", "bill_velocity",
    # volatility
    "bill_std", "payment_std", "bill_cv", "payment_cv",
    # interactions
    "age_limit_interaction", "util_delay_interaction",
    "payment_gap_last_month", "payment_gap_ratio",
    # streak and weighted
    "consecutive_delay_streak", "weighted_pay_delay",
    # monthly pay ratios (NEW)
    "pay_ratio_1", "pay_ratio_2", "pay_ratio_3",
    "pay_ratio_4", "pay_ratio_5", "pay_ratio_6",
    # deterioration (NEW)
    "deterioration",
    # underpayment (NEW)
    "months_underpaid",
    # utilisation trend (NEW)
    "util_recent", "util_older", "util_trend",
    # squared utilisation (NEW)
    "credit_utilisation_sq"
]


def get_feature_columns():
    return NUMERIC_FEATURES + ENGINEERED_FEATURES, CATEGORICAL_FEATURES


# ─────────────────────────────────────────────
#  Sklearn preprocessor
# ─────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    numeric_cols, categorical_cols = get_feature_columns()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="drop")

    return preprocessor


# ─────────────────────────────────────────────
#  Public interface
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame):
    """Engineer features and return X, y ready for sklearn."""
    df = engineer_features(df)
    numeric_cols, categorical_cols = get_feature_columns()
    all_cols = numeric_cols + categorical_cols

    X = df[all_cols]
    y = df[TARGET]
    return X, y


def save_preprocessor(preprocessor, path: str = None):
    if path is None:
        path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved → {path}")
    return path


def load_preprocessor(path: str = None):
    if path is None:
        path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    return joblib.load(path)


if __name__ == "__main__":
    from ingest import load_and_prepare
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    df = load_and_prepare()
    X, y = build_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features ({len(X.columns)} total):")
    for col in X.columns:
        print(f"  {col}")
    print(f"\nTarget distribution:\n{y.value_counts(normalize=True)}")