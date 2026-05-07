import pandas as pd
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "credit.csv")

REQUIRED_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default.payment.next.month"
]


@dataclass
class ValidationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and do basic type coercion."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Please download the UCI Credit Card Default dataset and place it at data/raw/credit.csv\n"
            "Download from: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset"
        )

    df = pd.read_csv(path)

    # the dataset sometimes has an extra ID column — drop it
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # strip whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {path}")
    return df


def validate_data(df: pd.DataFrame) -> ValidationResult:
    """Run data quality checks. Returns a ValidationResult."""
    errors = []
    warnings = []

    # --- schema checks ---
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        return ValidationResult(
            passed=False, errors=errors, warnings=warnings,
            row_count=len(df), column_count=len(df.columns)
        )

    # --- null checks ---
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.05]
    if not high_null.empty:
        errors.append(f"High null rate (>5%) in columns: {high_null.to_dict()}")

    low_null = null_pct[(null_pct > 0) & (null_pct <= 0.05)]
    if not low_null.empty:
        warnings.append(f"Low null rate in columns (will be imputed): {low_null.to_dict()}")

    # --- range checks ---
    if df["AGE"].lt(18).any():
        errors.append(f"AGE contains {df['AGE'].lt(18).sum()} values below 18")

    if df["AGE"].gt(100).any():
        warnings.append(f"AGE contains {df['AGE'].gt(100).sum()} values above 100")

    if df["LIMIT_BAL"].lt(0).any():
        errors.append(f"LIMIT_BAL contains {df['LIMIT_BAL'].lt(0).sum()} negative values")

    # --- duplicate check ---
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate rows detected — will be dropped")

    # --- class balance check ---
    target_col = "default.payment.next.month"
    if target_col in df.columns:
        pos_rate = df[target_col].mean()
        if pos_rate < 0.05:
            warnings.append(f"Very low positive class rate: {pos_rate:.2%}")
        elif pos_rate > 0.5:
            warnings.append(f"High positive class rate: {pos_rate:.2%}")
        else:
            print(f"Class balance: {pos_rate:.2%} default rate (healthy)")

    # --- row count check ---
    if len(df) < 1000:
        errors.append(f"Dataset too small: {len(df)} rows. Minimum 1000 required.")

    passed = len(errors) == 0
    return ValidationResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        row_count=len(df),
        column_count=len(df.columns)
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps: drop duplicates, impute nulls, fix known issues."""
    df = df.copy()

    # drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} duplicate rows")

    # impute nulls in numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {null_count} nulls in {col} with median={median_val:.2f}")

    # education: values 0, 5, 6 are undocumented — map to 'other' (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # marriage: value 0 is undocumented — map to 'other' (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    print(f"Cleaned dataset: {len(df):,} rows remaining")
    return df


def load_and_prepare(path: str = DATA_PATH) -> pd.DataFrame:
    """Full pipeline: load → validate → clean."""
    df = load_data(path)

    result = validate_data(df)

    if result.warnings:
        print("\nValidation warnings:")
        for w in result.warnings:
            print(f"  WARNING: {w}")

    if not result.passed:
        print("\nValidation errors:")
        for e in result.errors:
            print(f"  ERROR: {e}")
        raise ValueError(f"Data validation failed with {len(result.errors)} error(s).")

    print("Data validation passed.")
    df = clean_data(df)
    return df


if __name__ == "__main__":
    df = load_and_prepare()
    print(df.describe())