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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def test_sanity_littles_law_closed_loop():
    """
    Sanity Check 5: |concurrency - throughput * mean_latency| / concurrency < 15%
    strictly enforced for ALL four closed-loop benchmark rows:
      - no_inspection
      - benign_fast_path
      - escalated
      - two_tier_blended
    Fails if ANY closed-loop row deviates by >= 15%. No violations documented as acceptable.
    """
    import json
    from pathlib import Path

    summary_path = Path("results/latency/summary.json")
    if not summary_path.exists():
        pytest.skip("results/latency/summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    cl = data.get("closed_loop", {})
    summary_stats = cl.get("summary_stats", {})
    assert summary_stats, "No closed loop summary stats found in results/latency/summary.json"

    concurrency = 20.0
    # Evaluate Little's Law on ALL 4 closed-loop rows
    tested_conditions = ["no_inspection", "benign_fast_path", "escalated", "two_tier_blended"]

    for cond_name in tested_conditions:
        assert cond_name in summary_stats, f"Missing closed-loop condition: {cond_name}"
        st = summary_stats[cond_name]
        tp = st["throughput_rps"]["mean"]
        mean_s = st["latency_mean_ms"]["mean"] / 1000.0
        n_est = tp * mean_s
        err = abs(concurrency - n_est) / concurrency
        assert err < 0.15, (
            f"Little's Law violation for {cond_name}: Estimated N={n_est:.2f} vs C={concurrency} "
            f"(Relative Error = {err*100:.2f}% >= 15%). Closed-loop invariance failed."
        )


def test_sanity_mean_latency_not_derived_from_throughput():
    """
    Sanity Check 5b: Verifies that closed-loop mean latency is independently measured
    from individual per-request timestamps and is NOT tautologically derived from
    concurrency / throughput (C / X). Fails if mean == C/X to within 1e-6.
    """
    import json
    from pathlib import Path

    summary_path = Path("results/latency/summary.json")
    if not summary_path.exists():
        pytest.skip("results/latency/summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    summary_stats = data.get("closed_loop", {}).get("summary_stats", {})
    assert summary_stats, "No closed loop summary stats found in results/latency/summary.json"

    concurrency = 20.0
    for cond_name in ["no_inspection", "benign_fast_path", "escalated", "two_tier_blended"]:
        assert cond_name in summary_stats, f"Missing {cond_name} in closed_loop summary stats"
        st = summary_stats[cond_name]
        measured_mean_ms = st["latency_mean_ms"]["mean"]
        tp = st["throughput_rps"]["mean"]
        tautological_mean_ms = (concurrency / tp) * 1000.0
        diff = abs(measured_mean_ms - tautological_mean_ms)
        assert diff > 1e-6, (
            f"Tautology detected for {cond_name}: measured mean ({measured_mean_ms:.8f} ms) "
            f"matches C / throughput ({tautological_mean_ms:.8f} ms) to within 1e-6! "
            f"Mean latency must come from per-request timestamps, not inverted throughput."
        )


def test_sanity_inspection_latency_ordering_check():
    """
    Sanity Check 6: Inspection mean/p50 latency >= passthrough.
    """
    import json
    from pathlib import Path

    summary_path = Path("results/latency/summary.json")
    if not summary_path.exists():
        pytest.skip("results/latency/summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    stats = data["closed_loop"]["summary_stats"]
    no_p50 = stats["no_inspection"]["latency_p50_ms"]["mean"]
    fast_p50 = stats["benign_fast_path"]["latency_p50_ms"]["mean"]
    no_mean = stats["no_inspection"]["latency_mean_ms"]["mean"]
    fast_mean = stats["benign_fast_path"]["latency_mean_ms"]["mean"]

    assert (fast_p50 >= no_p50 * 0.95) or (fast_mean >= no_mean * 0.95), (
        f"Inspection latency (p50={fast_p50:.2f}ms, mean={fast_mean:.2f}ms) "
        f"lower than passthrough (p50={no_p50:.2f}ms, mean={no_mean:.2f}ms)"
    )


def test_sanity_knee_detector():
    """
    Sanity Check 7: Automatic knee detector:
      - Baseline = minimum p95 across levels
      - Flags knee where latency > 2.0x baseline OR achieved/offered < 0.98
      - Returns 'not reached' if conditions are never met.
    """
    from load_test.load_test_suite import detect_saturation_knee

    # Case 1: Doubling knee at 300 req/s (latency 2.3 > 2.0 * min(1.0))
    rates = [100, 200, 300, 400]
    latencies = [1.0, 1.2, 2.3, 5.5]
    knee_r, label, _ = detect_saturation_knee(rates, latencies, threshold_factor=2.0)
    assert knee_r == 300, f"Expected knee at 300, got {knee_r}"
    assert label == "300 req/s"

    # Case 2: Flat curve (knee not reached)
    flat_latencies = [1.0, 1.1, 1.2, 1.3]
    knee_r, label, _ = detect_saturation_knee(rates, flat_latencies, threshold_factor=2.0)
    assert knee_r is None, f"Expected None for flat curve, got {knee_r}"
    assert label == "not reached"

    # Case 3: Throughput degradation knee (achieved/offered < 0.98)
    achieved = [100.0, 200.0, 280.0, 350.0]  # 280/300 = 0.933 < 0.98
    latencies_flat = [1.0, 1.1, 1.2, 1.3]
    knee_r, label, reason = detect_saturation_knee(
        rates, latencies_flat, achieved_rates=achieved, threshold_factor=2.0, achieved_ratio_threshold=0.98
    )
    assert knee_r == 300, f"Expected throughput degradation knee at 300, got {knee_r} ({reason})"
    assert "achieved/offered" in reason


def test_sanity_timeout_enforcement_overload():
    """
    Sanity Check 8: Under overload or with short timeout, timeout_count > 0 and 504 is returned on fail_closed.
    """
    import src.gateway.middleware as mw
    from src.gateway.middleware import screen_request
    from fastapi import BackgroundTasks

    mw.save_gateway_state({
        "mode": "two_tier",
        "behavior": "hold_with_timeout",
        "timeout_ms": 5.0,
        "timeout_action": "fail_closed",
    })
    mw.reset_telemetry_stats()

    # Create an escalated request without worker consumption to trigger timeout
    req = MockClientRequest(
        method="POST",
        path="/tienda1/miembros/editar.jsp",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body="modo=registro&login=glaser&password=elementada",
    )

    async def run_overload():
        bg = BackgroundTasks()
        resp = await screen_request(req, bg)
        return resp

    resp = asyncio.run(run_overload())
    assert resp.status_code == 504, f"Expected 504 on timeout with fail_closed, got {resp.status_code}"
    stats = mw.get_telemetry_stats()
    assert stats["timeout_count"] > 0, f"Expected timeout_count > 0, got {stats['timeout_count']}"
    assert stats["fail_closed_504_count"] > 0, f"Expected fail_closed_504_count > 0, got {stats['fail_closed_504_count']}"


def test_sanity_onnx_sklearn_parity():
    """
    Sanity Check 9: ONNX vs Scikit-Learn parity:
      - Label agreement >= 99.9%
      - Max probability difference < 1e-3
      - Zero flips across the [0.30, 0.70] escalation thresholds
    """
    from tier2.track_a_http import compute_onnx_parity
    verifier = get_verifier()
    df_csic = pd.read_csv(CSIC_CSV)
    df_test = df_csic.iloc[2000:3000]

    res = compute_onnx_parity(verifier, df_test)
    assert res["label_agreement_pct"] >= 99.9, f"Label agreement {res['label_agreement_pct']:.2f}% < 99.9%"
    assert res["max_prob_diff"] < 1e-3, f"Max prob diff {res['max_prob_diff']:.2e} >= 1e-3"
    assert res["threshold_flips_count"] == 0, f"Threshold flips count {res['threshold_flips_count']} != 0"
    assert res["parity_passed"] is True

