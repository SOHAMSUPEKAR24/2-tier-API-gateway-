"""
src/utils/feature_guard.py
──────────────────────────
Enforces the feature allowlist (Project Rule 4).

Usage:
    from src.utils.feature_guard import enforce_allowlist, TIER1_ALLOWLIST
    X = enforce_allowlist(df, TIER1_ALLOWLIST)

The guard:
  1. Verifies no identifier column appears in the feature set.
  2. Returns only allowed columns in the canonical order.
  3. Raises ValueError loudly if an identifier column is present.
"""
from typing import List

import numpy as np
import pandas as pd

# ── Canonical ordered allowlists ───────────────────────────────────────────────
FEATURES_9: List[str] = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "ACK Flag Count",
    "PSH Flag Count",
]

TIER1_ALLOWLIST: List[str] = FEATURES_9

FEATURES_25: List[str] = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "ACK Flag Count",
    "PSH Flag Count",
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Fwd Packet Length Std",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Packet Length Mean",
    "Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Fwd IAT Max",
    "Bwd IAT Max",
    "SYN Flag Count",
    "RST Flag Count",
]

ALL_ALLOWLISTED_CIC_FEATURES: List[str] = [
    "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Packet Length Max", "Packet Length Min", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "Fwd Header Length", "Bwd Header Length", "Fwd Seg Size Min", "Fwd Act Data Pkts",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Flow IAT Std",
    "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Min", "Fwd IAT Mean", "Fwd IAT Std",
    "Bwd IAT Total", "Bwd IAT Max", "Bwd IAT Min", "Bwd IAT Mean", "Bwd IAT Std",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]


# Columns that are STRICTLY FORBIDDEN from appearing in any feature matrix
IDENTIFIER_COLUMNS: List[str] = [
    "Dst Port",
    "Protocol",
    "Flow ID",
    "Timestamp",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
]


def enforce_allowlist(
    df: pd.DataFrame,
    allowlist: List[str] = TIER1_ALLOWLIST,
    drop_identifiers: bool = True,
) -> np.ndarray:
    """
    Extract and validate features from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that may contain many columns (79 from CIC-IDS).
    allowlist : list of str
        Column names to extract. Order is canonical.
    drop_identifiers : bool, default True
        If True, safely strips identifier columns from df before feature extraction.
        If False and identifier columns are present, raises ValueError.

    Returns
    -------
    np.ndarray of shape (n_samples, len(allowlist))

    Raises
    ------
    ValueError if any identifier column is requested in allowlist.
    ValueError if drop_identifiers=False and an identifier column is found in df.
    ValueError if any allowlist column is missing from df.
    """
    # Strip whitespace from all column names for robustness
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # 1. Strictly forbid identifier columns from appearing in the requested allowlist
    forbidden_in_allowlist = [c for c in allowlist if c in IDENTIFIER_COLUMNS]
    if forbidden_in_allowlist:
        raise ValueError(
            f"[FeatureGuard] IDENTIFIER COLUMNS FORBIDDEN IN FEATURE SET: {forbidden_in_allowlist}"
        )

    # 2. Check for forbidden identifier columns in df
    found_identifiers = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
    if found_identifiers:
        if not drop_identifiers:
            raise ValueError(
                f"[FeatureGuard] IDENTIFIER COLUMNS DETECTED in input DataFrame: {found_identifiers}."
            )
        df = strip_identifiers(df)

    # 3. Verify all allowlisted columns exist
    missing = [col for col in allowlist if col not in df.columns]
    if missing:
        raise ValueError(
            f"[FeatureGuard] Missing allowlisted feature columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # 4. Extract only the allowed columns in canonical order
    X = df[allowlist].copy()

    # 5. Handle inf / NaN (common in CIC-IDS flow stats)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    return X.values


def strip_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with all identifier columns dropped.
    Safe to call even if identifier columns are absent.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def assert_no_leakage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    tol: float = 1e-8,
) -> None:
    """
    Assert there are no duplicate rows between train and test sets.
    Raises AssertionError if leakage is detected.
    """
    # Convert to tuple-of-tuples for set comparison
    train_set = set(map(tuple, X_train.tolist()))
    test_set = set(map(tuple, X_test.tolist()))
    overlap = train_set & test_set
    if len(overlap) > 0:
        raise AssertionError(
            f"[FeatureGuard] LEAKAGE DETECTED: {len(overlap)} identical rows "
            f"appear in both train and test splits."
        )
