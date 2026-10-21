"""
tests/test_tier2_tracks.py
──────────────────────────
Unit tests for Track A (HTTP) and Track B (PE) deep verification verifiers
and Pareto curve generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from tier2.track_a_http import (
    TrackAHTTPVerifier,
    extract_raw_http_text,
    extract_track_a_fast_features,
)
from tier2.track_b_pe import (
    TrackBPEVerifier,
    extract_track_b_tier1_features,
    extract_track_b_tier2_features,
)


@pytest.fixture
def synthetic_track_a_df():
    """Generates synthetic HTTP dataframe mimicking CSIC 2010 schema."""
    rows = []
    for i in range(120):
        is_attack = i % 2
        method = "POST" if is_attack else "GET"
        url = f"/login.php?user=admin' OR 1=1--&id={i}" if is_attack else f"/index.html?page={i}"
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        content = "payload=<script>alert(1)</script>" if is_attack else "param=normal_data"
        rows.append({
            "Method": method,
            "URL": url,
            "User-Agent": ua,
            "content": content,
            "classification": is_attack,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_track_b_records():
    """Generates synthetic PE records mimicking EMBER schema."""
    records = []
    rng = np.random.RandomState(42)
    for i in range(100):
        is_malware = i % 2
        record = {
            "general": {
                "size": 50000 + i * 100,
                "vsize": 60000 + i * 120,
                "has_signature": 0 if is_malware else 1,
                "has_tls": 1 if is_malware else 0,
            },
            "section": {
                "sections": [{"entropy": 7.5 if is_malware else 4.2}, {"entropy": 6.8 if is_malware else 3.5}]
            },
            "histogram": list(rng.randint(0, 100, 256).astype(float)),
            "byteentropy": list(rng.randint(0, 100, 256).astype(float)),
            "strings": {
                "numstrings": 200 if is_malware else 50,
                "avglength": 8.5,
                "printables": 150,
                "entropy": 5.2,
                "paths": 3,
                "urls": 2,
                "registry": 1,
                "MZ": 1,
            },
            "label": is_malware,
        }
        records.append(record)
    return records


def test_track_a_pareto_monotonicity(synthetic_track_a_df):
    """Asserts that Track A Pareto sweep satisfies monotonic budget scaling."""
    df = synthetic_track_a_df
    split = 80
    df_tr, df_te = df.iloc[:split], df.iloc[split:]

    verifier = TrackAHTTPVerifier(random_state=42)
    verifier.fit(df_tr)

    budgets = [0.0, 0.10, 0.25, 0.50, 1.0]
    pareto = verifier.evaluate_pareto(df_te, budgets=budgets)

    # 1. Check all budgets present
    for b in budgets:
        assert b in pareto

    # 2. At budget 0%, ER = 0% and accuracy equals Tier 1 alone
    assert pareto[0.0]["escalation_rate"] == 0.0
    assert pareto[0.0]["two_tier_accuracy"] == pareto[0.0]["tier1_only_accuracy"]

    # 3. At budget 100%, ER = 100% and accuracy equals Tier 2 alone
    assert pareto[1.0]["escalation_rate"] == 100.0
    assert pareto[1.0]["two_tier_accuracy"] == pareto[1.0]["tier2_only_accuracy"]
    assert pareto[1.0]["silent_failure_rate"] == 0.0

    # 4. Latency is non-decreasing with escalation rate
    prev_lat = pareto[0.0]["blended_latency_p50_ms"]
    for b in budgets[1:]:
        curr_lat = pareto[b]["blended_latency_p50_ms"]
        assert curr_lat >= prev_lat - 1e-4
        prev_lat = curr_lat

    # 5. Silent-failure rate is non-increasing with budget
    prev_sfr = pareto[0.0]["silent_failure_rate"]
    for b in budgets[1:]:
        curr_sfr = pareto[b]["silent_failure_rate"]
        assert curr_sfr <= prev_sfr + 1e-4
        prev_sfr = curr_sfr


def test_track_b_pareto_monotonicity(synthetic_track_b_records):
    """Asserts that Track B Pareto sweep satisfies monotonic budget scaling."""
    records = synthetic_track_b_records
    split = 70
    tr, te = records[:split], records[split:]

    verifier = TrackBPEVerifier(random_state=42)
    verifier.fit(tr)

    budgets = [0.0, 0.20, 0.50, 1.0]
    pareto = verifier.evaluate_pareto(te, budgets=budgets)

    assert pareto[0.0]["escalation_rate"] == 0.0
    assert pareto[0.0]["two_tier_accuracy"] == pareto[0.0]["tier1_only_accuracy"]

    assert pareto[1.0]["escalation_rate"] == 100.0
    assert pareto[1.0]["two_tier_accuracy"] == pareto[1.0]["tier2_only_accuracy"]
    assert pareto[1.0]["silent_failure_rate"] == 0.0
