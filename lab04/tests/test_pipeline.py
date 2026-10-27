"""
Automated Unit and Integration Tests for Lab 04 Pipeline.
"""

import sys
from pathlib import Path

# Add lab04 root to sys.path so 'src' can be imported cleanly
LAB04_ROOT = Path(__file__).resolve().parent.parent
if str(LAB04_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB04_ROOT))

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset import ingest_raw_data
from src.feature_engineering import compute_derived_features, AdultFeatureEngineer
from src.label_audit import run_label_quality_audit
from src.modeling import prepare_features_and_targets
from src.preprocessing import (
    NumericOutlierClipper,
    build_preprocessor,
    deduplicate_train_only,
    load_preprocessor,
    perform_stratified_split,
    save_preprocessor,
)
from src.validation import validate_raw_data

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def raw_data(config):
    df, meta = ingest_raw_data(config=config, base_dir=BASE_DIR, force_download=False)
    return df, meta


def test_raw_data_dimensions(raw_data):
    """Verify raw data dimensions match exactly 48,842 rows and 15 columns."""
    df, meta = raw_data
    assert df.shape == (48842, 15)
    assert meta["rows"] == 48842
    assert meta["columns"] == 15
    assert "income" in df.columns
    assert len(meta["feature_names"]) == 14


def test_validation_schema_and_ranges(raw_data, config):
    """Verify schema, types, range checks, and duplicate detection."""
    df, _ = raw_data
    report, is_valid = validate_raw_data(df, config, base_dir=BASE_DIR)
    assert is_valid is True
    assert report["summary"]["total_rows"] == 48842
    assert report["summary"]["total_columns"] == 15

    # Check numeric ranges passed
    for col, r_res in report["range_validation"].items():
        assert r_res["range_valid"] is True, f"Range violation in {col}"
        assert r_res["total_violations"] == 0

    # Check duplicate records reported
    assert report["duplicates"]["exact_duplicate_rows"] == 52


def test_stratified_split_and_no_leakage(raw_data, config):
    """Verify 70/30 split, target stratification, and test set integrity."""
    df, _ = raw_data
    train_df, test_df = perform_stratified_split(df, config)

    # Check split sizes
    assert len(train_df) == 34189
    assert len(test_df) == 14653
    assert len(train_df) + len(test_df) == 48842

    # Verify target proportions match closely (within 0.05%)
    raw_prop = (df["income"].astype(str) == ">50K").mean()
    train_prop = (train_df["income"].astype(str) == ">50K").mean()
    test_prop = (test_df["income"].astype(str) == ">50K").mean()

    assert abs(raw_prop - train_prop) < 0.001
    assert abs(raw_prop - test_prop) < 0.001

    # Verify deduplication only touches train set
    train_dedup, dedup_meta = deduplicate_train_only(train_df, target_col="income")
    assert dedup_meta["exact_duplicates_removed"] == 24
    assert len(train_dedup) == 34165
    assert len(test_df) == 14653  # Test set remains untouched


def test_target_leakage_prevention(raw_data, config):
    """Verify target column is completely removed from feature matrices."""
    df, _ = raw_data
    train_df, test_df = perform_stratified_split(df, config)
    train_dedup, _ = deduplicate_train_only(train_df, target_col="income")

    X_train, y_train, X_test, y_test = prepare_features_and_targets(
        train_dedup, test_df, target_col="income"
    )

    assert "income" not in X_train.columns
    assert "income" not in X_test.columns
    assert "class" not in X_train.columns
    assert "class" not in X_test.columns
    assert len(X_train.columns) == 14
    assert len(X_test.columns) == 14
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_numeric_outlier_clipper():
    """Verify NumericOutlierClipper fits on train data only and clips correctly."""
    clipper = NumericOutlierClipper(lower_percentile=5.0, upper_percentile=95.0)
    train_data = np.arange(1, 101, dtype=float).reshape(-1, 1)  # 1 to 100
    clipper.fit(train_data)

    low, high = clipper.bounds_[0]
    assert low == pytest.approx(5.95, 0.1)
    assert high == pytest.approx(95.05, 0.1)

    # Test data with extreme values
    test_data = np.array([[-100.0], [50.0], [1000.0]])
    clipped = clipper.transform(test_data)

    assert clipped[0, 0] == pytest.approx(low)
    assert clipped[1, 0] == pytest.approx(50.0)
    assert clipped[2, 0] == pytest.approx(high)


def test_feature_engineering_safety():
    """Verify safe division and calculation in feature engineering."""
    sample_df = pd.DataFrame({
        "capital-gain": [1000, 0, 500],
        "capital-loss": [200, 100, 0],
        "hours-per-week": [40, 20, 50],
        "age": [36, 16, np.nan]  # Age 16 triggers division safeguard
    })

    fe_df = compute_derived_features(sample_df, age_offset=16.0, min_denominator=1.0)

    # Check capital_net
    assert list(fe_df["capital_net"]) == [800.0, -100.0, 500.0]

    # Check hours_per_age
    # Row 0: 40 / (36 - 16) = 40 / 20 = 2.0
    assert fe_df.loc[0, "hours_per_age"] == pytest.approx(2.0)
    # Row 1: age 16 -> denom <= 0 -> clamped to 1.0 -> 20 / 1.0 = 20.0
    assert fe_df.loc[1, "hours_per_age"] == pytest.approx(20.0)
    # Row 2: age NaN -> clamped to 1.0 -> 50 / 1.0 = 50.0
    assert fe_df.loc[2, "hours_per_age"] == pytest.approx(50.0)


def test_preprocessor_serialization(raw_data, config, tmp_path):
    """Verify preprocessor can be saved and loaded with joblib without degradation."""
    df, _ = raw_data
    train_df, test_df = perform_stratified_split(df, config)
    train_dedup, _ = deduplicate_train_only(train_df, target_col="income")

    base_num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    X_train = train_dedup.drop(columns=["income"])
    preprocessor = build_preprocessor(base_num_cols, cat_cols, config)
    X_train_trans = preprocessor.fit_transform(X_train)

    joblib_path = tmp_path / "test_preprocessor.joblib"
    save_preprocessor(preprocessor, joblib_path)
    # Test loading with Path object
    loaded_preprocessor = load_preprocessor(joblib_path)
    # Test loading with string path
    loaded_str = load_preprocessor(str(joblib_path))

    X_train_trans_reloaded = loaded_preprocessor.transform(X_train)
    X_train_trans_str = loaded_str.transform(X_train)
    np.testing.assert_allclose(X_train_trans, X_train_trans_reloaded, rtol=1e-5)
    np.testing.assert_allclose(X_train_trans, X_train_trans_str, rtol=1e-5)

    # Test loading the actual pipeline preprocessor.joblib from results/
    actual_joblib = BASE_DIR / "results" / "preprocessor.joblib"
    if actual_joblib.exists():
        loaded_actual = load_preprocessor(str(actual_joblib))
        assert loaded_actual is not None


def test_base_dir_string_normalization(config):
    """Verify that ingest_raw_data accepts base_dir as a string as well as Path."""
    df, meta = ingest_raw_data(config=config, base_dir=str(BASE_DIR), force_download=False)
    assert df.shape == (48842, 15)
    assert meta["rows"] == 48842


def test_label_quality_audit(raw_data, config):
    """Verify reproducible 50-row sampling and duplicate feature conflicting label audit."""
    df, _ = raw_data
    sampled_50, conflicting_rows, summary = run_label_quality_audit(df, config, base_dir=BASE_DIR)

    assert len(sampled_50) == 50
    assert summary["sample_size"] == 50
    assert summary["total_conflicting_rows"] == 10
    assert summary["unique_conflicting_feature_vectors"] == 5
    assert len(conflicting_rows) == 10
