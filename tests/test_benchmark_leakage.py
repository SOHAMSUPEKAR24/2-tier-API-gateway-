"""
tests/test_benchmark_leakage.py
───────────────────────────────
Unit tests for data integrity and leakage prevention (Project Rules 1, 3, 4).
Asserts:
  1. No identifier columns appear in the feature sets or allowlists.
  2. Feature guard strictly rejects forbidden columns.
  3. Zero duplicate rows exist between train and test splits (zero data leakage).
  4. Benchmark splits handle +/- inf and NaN without crashing.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_guard import (
    IDENTIFIER_COLUMNS,
    TIER1_ALLOWLIST,
    assert_no_leakage,
    enforce_allowlist,
    strip_identifiers,
)


def test_allowlist_contains_no_identifiers():
    """Ensure the canonical TIER1_ALLOWLIST contains ZERO forbidden identifier columns."""
    overlap = set(TIER1_ALLOWLIST) & set(IDENTIFIER_COLUMNS)
    assert len(overlap) == 0, f"TIER1_ALLOWLIST contains forbidden identifiers: {overlap}"


def test_feature_guard_rejects_identifier_in_allowlist():
    """Verify that requesting an identifier in allowlist raises ValueError."""
    import pandas as pd

    df = pd.DataFrame({
        "Dst Port": [80, 443],
        "Flow Bytes/s": [100.0, 200.0],
    })
    with pytest.raises(ValueError, match="IDENTIFIER COLUMNS FORBIDDEN"):
        enforce_allowlist(df, allowlist=["Dst Port", "Flow Bytes/s"])


def test_assert_no_leakage_detects_overlap():
    """Verify that assert_no_leakage catches duplicate rows between train and test."""
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test_leaky = np.array([[3.0, 4.0], [7.0, 8.0]])  # Row [3.0, 4.0] is in train!
    X_test_clean = np.array([[9.0, 10.0], [11.0, 12.0]])

    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        assert_no_leakage(X_train, X_test_leaky)

    # Clean splits must pass without raising
    assert_no_leakage(X_train, X_test_clean)


def test_benchmark_splits_leakage_and_shape():
    """
    If benchmark dataset exists, load it and verify:
      1. Zero train-test leakage across all protocols.
      2. Exactly 9 features in every matrix.
      3. No NaNs or infinities.
    """
    npz_path = "data/processed/benchmark_dataset.npz"
    if not os.path.exists(npz_path):
        pytest.skip(f"Benchmark splits not yet generated at {npz_path}")

    data = np.load(npz_path)
    protocols = set([k[:-8] for k in data.files if k.endswith("_X_train")])

    for proto in protocols:
        X_tr = data[f"{proto}_X_train"]
        y_tr = data[f"{proto}_y_train"]
        X_te = data[f"{proto}_X_test"]
        y_te = data[f"{proto}_y_test"]

        # 1. Feature count must equal canonical allowlist length (9)
        assert X_tr.shape[1] == len(TIER1_ALLOWLIST), f"Train features in {proto} != {len(TIER1_ALLOWLIST)}"
        assert X_te.shape[1] == len(TIER1_ALLOWLIST), f"Test features in {proto} != {len(TIER1_ALLOWLIST)}"

        # Verify 25 and 76 feature matrices if present
        if f"{proto}_X_train_25" in data:
            assert data[f"{proto}_X_train_25"].shape[1] == 25
            assert data[f"{proto}_X_test_25"].shape[1] == 25
            assert np.all(np.isfinite(data[f"{proto}_X_train_25"]))
            assert np.all(np.isfinite(data[f"{proto}_X_test_25"]))

        if f"{proto}_X_train_76" in data:
            assert data[f"{proto}_X_train_76"].shape[1] == 76
            assert data[f"{proto}_X_test_76"].shape[1] == 76
            assert np.all(np.isfinite(data[f"{proto}_X_train_76"]))
            assert np.all(np.isfinite(data[f"{proto}_X_test_76"]))
            assert_no_leakage(data[f"{proto}_X_train_76"], data[f"{proto}_X_test_76"])

        # 2. No NaNs or Infs
        assert np.all(np.isfinite(X_tr)), f"NaN/Inf found in {proto} train set"
        assert np.all(np.isfinite(X_te)), f"NaN/Inf found in {proto} test set"

        # 3. Zero train-test row leakage
        assert_no_leakage(X_tr, X_te)

        # 4. Binary targets only
        assert set(np.unique(y_tr)).issubset({0, 1}), f"Non-binary target in {proto} train"
        assert set(np.unique(y_te)).issubset({0, 1}), f"Non-binary target in {proto} test"



def test_benchmark_stats_json_validity():
    """Verify that results/benchmark_stats.json exists and adheres to schema."""
    stats_path = Path("results/benchmark_stats.json")
    if not stats_path.exists():
        pytest.skip("benchmark_stats.json not yet written")

    with open(stats_path) as f:
        stats = json.load(f)

    assert "dataset_name" in stats
    assert "known_label_issues" in stats
    assert "features" in stats
    assert stats["feature_count"] == len(TIER1_ALLOWLIST)
    assert "protocols" in stats
    assert "ftp_only" in stats["protocols"]
