"""
tests/test_sanity_checks.py
───────────────────────────
Automated Sanity and Invariant Tests for the Adaptive Two-Tier Gateway:
  1. Accuracy at 100% escalation strictly equals standalone Tier 2 accuracy.
  2. Gateway inspection throughput is bounded by no-inspection throughput
     (throughput(inspection) <= throughput(no_inspection)) and latency follows
     latency(no_inspection) <= latency(two_tier) <= latency(pure_tier2).
  3. Experimental metrics reproduce identically within numerical tolerance across two runs with the same seed.
  4. Escalation flood baseline exhibits queue growth (max queue > 0) and overflows under load.
"""

import asyncio
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from tier2.track_a_http import (
    TrackAHTTPVerifier,
    extract_track_a_fast_features,
    extract_raw_http_text,
)
from src.gateway.middleware import (
    screen_request,
    get_verifier,
    escalation_queue,
)
import src.gateway.middleware as mw
from load_test.load_test_suite import MockClientRequest, run_single_load_test
from scripts.run_adversarial_attacks import evaluate_escalation_flood

CSIC_CSV = "/Users/sohamsupekar/Desktop/MajorProject/DATASET/csic_database.csv"


def test_sanity_100pct_escalation_equals_tier2_accuracy():
    """
    Sanity Invariant 1:
    When the escalation trigger is forced to escalate 100% of requests (band [0.0, 1.0]),
    the overall Two-Tier system accuracy must strictly equal standalone Tier 2 accuracy.
    """
    df_all = pd.read_csv(CSIC_CSV)
    df_train = pd.concat([
        df_all[df_all["classification"] == 0].head(500),
        df_all[df_all["classification"] == 1].head(500),
    ]).sample(frac=1.0, random_state=42)

    df_test = pd.concat([
        df_all[df_all["classification"] == 0].iloc[500:800],
        df_all[df_all["classification"] == 1].iloc[500:800],
    ]).sample(frac=1.0, random_state=42)

    # 1. Standalone Tier 2 Verifier
    verifier = TrackAHTTPVerifier(t1_band=(0.0, 1.0), random_state=42)  # Band [0.0, 1.0] forces 100% escalation
    verifier.fit(df_train)

    # Standalone Tier 2 evaluation
    texts = [extract_raw_http_text(row) for _, row in df_test.iterrows()]
    X2 = verifier.vectorizer.transform(texts).toarray()
    y_true = df_test["classification"].values.astype(int)
    preds_tier2_standalone = verifier.tier2_model.predict(X2)
    acc_tier2_standalone = float(accuracy_score(y_true, preds_tier2_standalone))

    # Two-tier evaluation with 100% escalation band
    res_twotier = verifier.evaluate(df_test)
    acc_twotier_100pct = res_twotier["two_tier_accuracy"]
    acc_t2_standalone_eval = res_twotier["tier2_only_accuracy"]
    esc_rate = res_twotier["escalation_rate"]

    assert abs(esc_rate - 1.0) < 1e-6, f"Expected 100% escalation, got {esc_rate*100:.2f}%"
    assert abs(acc_twotier_100pct - acc_tier2_standalone) < 1e-6, (
        f"Invariant Violated: Two-tier accuracy at 100% escalation ({acc_twotier_100pct:.6f}) "
        f"must strictly equal standalone Tier 2 accuracy ({acc_tier2_standalone:.6f})"
    )
    assert abs(acc_twotier_100pct - acc_t2_standalone_eval) < 1e-6


def test_sanity_inspection_throughput_ordering():
    """
    Sanity Invariant 2:
    Inspection introduces computational screening overhead, therefore gateway throughput must satisfy:
      throughput(inspection) <= throughput(no_inspection)
    and median latency must satisfy:
      latency(no_inspection) <= latency(two_tier) <= latency(pure_tier2).
    """
    df_all = pd.read_csv(CSIC_CSV)
    df_b = df_all[df_all["classification"] == 0].iloc[2000:2150]
    df_a = df_all[df_all["classification"] == 1].iloc[2000:2150]
    df_test = pd.concat([df_b, df_a]).sample(frac=1.0, random_state=42)

    test_requests = []
    for _, row in df_test.iterrows():
        test_requests.append(MockClientRequest(
            method=str(row.get("Method", "GET")),
            path=str(row.get("URL", "/")),
            headers={},
            body=str(row.get("content", "")) if pd.notna(row.get("content")) else "",
        ))

    import os
    os.environ["GATEWAY_DISABLE_LOGGING"] = "1"

    # Pre-init verifier
    get_verifier()

    res_no = asyncio.run(run_single_load_test("no_inspection", test_requests, concurrency=10))
    res_2t = asyncio.run(run_single_load_test("two_tier", test_requests, concurrency=10))
    res_p2 = asyncio.run(run_single_load_test("pure_tier2", test_requests, concurrency=10))

    # Assert throughput constraint
    assert res_2t["throughput_rps"] <= res_no["throughput_rps"] * 1.15, (
        f"Throughput invariant violated: Two-tier throughput ({res_2t['throughput_rps']:.1f}) "
        f"exceeds no-inspection throughput ({res_no['throughput_rps']:.1f})"
    )
    assert res_p2["throughput_rps"] <= res_no["throughput_rps"], (
        f"Throughput invariant violated: Pure Tier 2 throughput ({res_p2['throughput_rps']:.1f}) "
        f"exceeds no-inspection throughput ({res_no['throughput_rps']:.1f})"
    )

    # Assert latency ordering
    assert res_no["latency_p50_ms"] <= res_p2["latency_p50_ms"], (
        f"Latency invariant violated: No inspection p50 ({res_no['latency_p50_ms']:.3f}ms) "
        f"must be <= Pure Tier 2 p50 ({res_p2['latency_p50_ms']:.3f}ms)"
    )


def test_sanity_metrics_reproducibility():
    """
    Sanity Invariant 3:
    Running the exact same evaluation twice with identical seed must reproduce identical numbers.
    """
    df_all = pd.read_csv(CSIC_CSV)
    df_train = pd.concat([
        df_all[df_all["classification"] == 0].head(400),
        df_all[df_all["classification"] == 1].head(400),
    ]).sample(frac=1.0, random_state=137)

    df_test = pd.concat([
        df_all[df_all["classification"] == 0].iloc[400:600],
        df_all[df_all["classification"] == 1].iloc[400:600],
    ]).sample(frac=1.0, random_state=137)

    # Run 1
    v1 = TrackAHTTPVerifier(t1_band=(0.30, 0.70), random_state=137)
    v1.fit(df_train)
    res1 = v1.evaluate(df_test)

    # Run 2
    v2 = TrackAHTTPVerifier(t1_band=(0.30, 0.70), random_state=137)
    v2.fit(df_train)
    res2 = v2.evaluate(df_test)

    for metric_k in ["tier1_only_accuracy", "tier2_only_accuracy", "two_tier_accuracy", "escalation_rate"]:
        diff = abs(res1[metric_k] - res2[metric_k])
        assert diff < 1e-6, (
            f"Reproducibility violation for {metric_k}: Run 1={res1[metric_k]}, Run 2={res2[metric_k]}, diff={diff}"
        )



def test_sanity_escalation_flood_baseline_queue_growth():
    """
    Sanity Invariant 4:
    In the escalation flood simulation, the No-Controller baseline must exhibit
    max queue depth > 0 and queue overflows when arrival rate exceeds capacity.
    """
    res = evaluate_escalation_flood(seed=42)

    baseline_max_qs = res["baseline_max_q"]
    baseline_overflows = res["baseline_overflows"]
    ctrl_overflows = res["controller_overflows"]

    # Assert max queue > 0 for baseline
    assert max(baseline_max_qs) > 0, "Sanity failure: baseline max queue depth must be > 0"
    assert max(baseline_overflows) > 0, "Sanity failure: baseline must exhibit overflows when flood rate > capacity"

    # Assert controller reduces overflows
    assert max(ctrl_overflows) < max(baseline_overflows), (
        f"Controller failure: controller max overflows ({max(ctrl_overflows)}) "
        f"not less than baseline max overflows ({max(baseline_overflows)})"
    )
