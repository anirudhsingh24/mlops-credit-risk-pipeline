import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import engineer_features, build_preprocessor, get_feature_columns, ENGINEERED_FEATURES
from ingest import validate_data, clean_data


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_row():
    """A single realistic customer record."""
    return {
        "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 10000, "BILL_AMT2": 9500, "BILL_AMT3": 9000,
        "BILL_AMT4": 8500,  "BILL_AMT5": 8000, "BILL_AMT6": 7500,
        "PAY_AMT1": 5000,  "PAY_AMT2": 4500,  "PAY_AMT3": 4000,
        "PAY_AMT4": 3500,  "PAY_AMT5": 3000,  "PAY_AMT6": 2500,
        "default.payment.next.month": 0
    }


@pytest.fixture
def sample_df(sample_row):
    return pd.DataFrame([sample_row] * 1100)


@pytest.fixture
def high_risk_row():
    """A customer with clear high-risk signals."""
    return {
        "LIMIT_BAL": 10000, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 2, "AGE": 25,
        "PAY_0": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 1, "PAY_5": 1, "PAY_6": 0,
        "BILL_AMT1": 9800, "BILL_AMT2": 9600, "BILL_AMT3": 9400,
        "BILL_AMT4": 9200,  "BILL_AMT5": 9000, "BILL_AMT6": 8800,
        "PAY_AMT1": 100, "PAY_AMT2": 100, "PAY_AMT3": 100,
        "PAY_AMT4": 100, "PAY_AMT5": 100, "PAY_AMT6": 100,
        "default.payment.next.month": 1
    }


# ─────────────────────────────────────────────
#  Feature engineering tests
# ─────────────────────────────────────────────

class TestFeatureEngineering:

    def test_all_engineered_features_created(self, sample_df):
        result = engineer_features(sample_df)
        for feat in ENGINEERED_FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_repayment_ratio_range(self, sample_df):
        result = engineer_features(sample_df)
        assert result["repayment_ratio"].between(0, 10).all(), \
            "repayment_ratio out of expected range"

    def test_credit_utilisation_clipped(self, sample_df):
        result = engineer_features(sample_df)
        assert result["credit_utilisation"].le(2).all(), \
            "credit_utilisation should be clipped at 2"
        assert result["credit_utilisation"].ge(0).all(), \
            "credit_utilisation should be >= 0"

    def test_no_division_by_zero(self):
        """Customer with zero bill amounts — should not crash."""
        zero_bill_row = {
            "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
            "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
            "BILL_AMT1": 0, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
            "default.payment.next.month": 0
        }
        df = pd.DataFrame([zero_bill_row])
        result = engineer_features(df)
        assert not result.isnull().any().any(), "NaN values in features with zero bills"
        assert not np.isinf(result.select_dtypes(include=np.number).values).any(), \
            "Inf values in features with zero bills"

    def test_high_risk_signals(self):
        """High risk customer should have higher risk feature values."""
        low_risk  = pd.DataFrame([{
            "LIMIT_BAL": 500000, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 45,
            "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
            "BILL_AMT1": 5000, "BILL_AMT2": 4800, "BILL_AMT3": 4600,
            "BILL_AMT4": 4400, "BILL_AMT5": 4200, "BILL_AMT6": 4000,
            "PAY_AMT1": 5000, "PAY_AMT2": 4800, "PAY_AMT3": 4600,
            "PAY_AMT4": 4400, "PAY_AMT5": 4200, "PAY_AMT6": 4000,
            "default.payment.next.month": 0
        }])
        high_risk = pd.DataFrame([{
            "LIMIT_BAL": 10000, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 2, "AGE": 22,
            "PAY_0": 3, "PAY_2": 3, "PAY_3": 2, "PAY_4": 2, "PAY_5": 1, "PAY_6": 1,
            "BILL_AMT1": 9900, "BILL_AMT2": 9800, "BILL_AMT3": 9700,
            "BILL_AMT4": 9600, "BILL_AMT5": 9500, "BILL_AMT6": 9400,
            "PAY_AMT1": 50, "PAY_AMT2": 50, "PAY_AMT3": 50,
            "PAY_AMT4": 50, "PAY_AMT5": 50, "PAY_AMT6": 50,
            "default.payment.next.month": 1
        }])

        lr = engineer_features(low_risk)
        hr = engineer_features(high_risk)

        assert hr["avg_pay_delay"].values[0] > lr["avg_pay_delay"].values[0]
        assert hr["credit_utilisation"].values[0] > lr["credit_utilisation"].values[0]
        assert hr["repayment_ratio"].values[0] < lr["repayment_ratio"].values[0]

    def test_consecutive_delay_streak(self):
        """Streak should count correctly from most recent month."""
        row = pd.DataFrame([{
            "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
            "PAY_0": 2, "PAY_2": 1, "PAY_3": 0, "PAY_4": 0, "PAY_5": 1, "PAY_6": 2,
            "BILL_AMT1": 1000, "BILL_AMT2": 1000, "BILL_AMT3": 1000,
            "BILL_AMT4": 1000, "BILL_AMT5": 1000, "BILL_AMT6": 1000,
            "PAY_AMT1": 100, "PAY_AMT2": 100, "PAY_AMT3": 100,
            "PAY_AMT4": 100, "PAY_AMT5": 100, "PAY_AMT6": 100,
            "default.payment.next.month": 0
        }])
        result = engineer_features(row)
        # PAY_0=2 (late), PAY_2=1 (late), PAY_3=0 (on time) → streak = 2
        assert result["consecutive_delay_streak"].values[0] == 2


# ─────────────────────────────────────────────
#  Preprocessor tests
# ─────────────────────────────────────────────

class TestPreprocessor:

    def test_preprocessor_output_shape(self, sample_df):
        from features import build_features
        X, y = build_features(sample_df)
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)
        assert X_proc.shape[0] == len(X), "Row count changed after preprocessing"
        assert X_proc.shape[1] >= X.shape[1], "Expected more columns after one-hot encoding"

    def test_no_nans_after_preprocessing(self, sample_df):
        from features import build_features
        X, y = build_features(sample_df)
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)
        assert not np.isnan(X_proc).any(), "NaN values after preprocessing"


# ─────────────────────────────────────────────
#  Validation tests
# ─────────────────────────────────────────────

class TestDataValidation:

    def test_valid_data_passes(self, sample_df):
        result = validate_data(sample_df)
        assert result.passed

    def test_missing_column_fails(self, sample_df):
        broken = sample_df.drop(columns=["LIMIT_BAL"])
        result = validate_data(broken)
        assert not result.passed
        assert any("LIMIT_BAL" in e for e in result.errors)

    def test_underage_customer_fails(self, sample_df):
        broken = sample_df.copy()
        broken.loc[0, "AGE"] = 15
        result = validate_data(broken)
        assert not result.passed

    def test_duplicate_drop(self, sample_df):
    # create a df with some unique rows and some duplicates
    unique_rows = pd.DataFrame([
        {**sample_row, "AGE": age} 
        for age, sample_row in [(25, sample_df.iloc[0].to_dict()),
                                 (30, sample_df.iloc[0].to_dict()),
                                 (35, sample_df.iloc[0].to_dict())]
    ])
    with_dupes = pd.concat([unique_rows, unique_rows])   # 6 rows, 3 unique
    cleaned    = clean_data(with_dupes)
    assert len(cleaned) == 3