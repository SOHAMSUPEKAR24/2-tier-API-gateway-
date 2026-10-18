"""
scripts/run_main_pipeline.py
─────────────────────────────
Reproducible main pipeline experiment.

Replaces: src/evaluation/run_real_pipeline.py
Fixes:
  - Runs across N seeds, reports mean ± 95% CI
  - Writes results to results/main_pipeline/<timestamp>_seed<s>/result.json
  - Uses feature_guard to enforce allowlist (no identifier columns)
  - Latency is MEASURED, not typed

Run with: make exp-main-pipeline
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_guard import TIER1_ALLOWLIST, assert_no_leakage, enforce_allowlist
from src.utils.result_writer import load_config, make_run_dir, mean_ci95, write_result


def measure_latency(model, X_single, n_warm=20, n_measure=200):
    """Measure per-sample inference latency in ms. Returns (median_ms, p95_ms, p99_ms)."""
    sample = X_single[:1]
    # Warmup
    for _ in range(n_warm):
        model.predict_proba(sample)
    # Measure
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(np.median(arr)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99))


def run_single_seed(seed: int, config: dict) -> dict:
    """Run the full pipeline for one seed. Returns a metrics dict."""
    print(f"\n  [Seed {seed}] Loading data...")

    # Load Tier 1 corpus
    df_t1 = pd.read_csv("data/generated/tier1_real_corpus.csv")
    df_t1.columns = [c.strip() for c in df_t1.columns]

    # FEATURE GUARD: drop identifiers, use only allowlist
    X_t1 = enforce_allowlist(df_t1, TIER1_ALLOWLIST)
    y = (df_t1["Label"] != "Benign").astype(int).values

    # Load Tier 2 corpus
    t2_records = []
    with open("data/generated/tier2_real_corpus.jsonl", "r") as f:
        for line in f:
            t2_records.append(json.loads(line))

    from src.features.real_feature_adapter import adapt_tier2_features
    X_t2 = adapt_tier2_features(t2_records)
    y_t2 = np.array([1 if r["label"] == 1 else 0 for r in t2_records])

    # Reproducible shuffle
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    X_t1, X_t2, y = X_t1[idx], X_t2[idx], y[idx]

    # Splits (same as original paper)
    cfg_c = config.get("corpus", {})
    test_sz = cfg_c.get("test_size", 2000)
    t2_train_sz = cfg_c.get("tier2_train_size", 3000)
    seed_sz = cfg_c.get("seed_size", 300)

    X_t1_test, y_test = X_t1[:test_sz], y[:test_sz]
    X_t2_train, y_t2_train = X_t2[test_sz:test_sz + t2_train_sz], y[test_sz:test_sz + t2_train_sz]
    X_t1_seed, y_seed = X_t1[test_sz + t2_train_sz:test_sz + t2_train_sz + seed_sz], y[test_sz + t2_train_sz:test_sz + t2_train_sz + seed_sz]
    X_t1_pool, X_t2_pool, y_pool = (
        X_t1[test_sz + t2_train_sz + seed_sz:],
        X_t2[test_sz + t2_train_sz + seed_sz:],
        y[test_sz + t2_train_sz + seed_sz:],
    )

    # Leakage check
    assert_no_leakage(X_t1_seed, X_t1_test)

    # Models
    cfg_t1 = config.get("tier1", {})
    cfg_t2 = config.get("tier2", {})
    esc_band = cfg_t1.get("escalation_band", [0.30, 0.70])

    tier1 = RandomForestClassifier(
        n_estimators=cfg_t1.get("n_estimators", 40),
        max_depth=cfg_t1.get("max_depth", 6),
        random_state=seed,
    )
    tier2 = RandomForestClassifier(
        n_estimators=cfg_t2.get("n_estimators", 250),
        random_state=seed,
    )

    tier2.fit(X_t2_train, y_t2_train)
    tier1.fit(X_t1_seed, y_seed)

    def evaluate(model, Xeval, yeval):
        preds = model.predict(Xeval)
        probs = model.predict_proba(Xeval)
        max_p = np.max(probs, axis=1)
        esc_mask = (max_p >= esc_band[0]) & (max_p <= esc_band[1])
        acc = float(accuracy_score(yeval, preds))
        f1 = float(f1_score(yeval, preds, zero_division=0))
        prec = float(precision_score(yeval, preds, zero_division=0))
        rec = float(recall_score(yeval, preds, zero_division=0))
        esc = float(np.mean(esc_mask))
        # Silent-failure rate: novel attacks classified confidently as benign
        confident_benign = (max_p > esc_band[1]) & (preds == 0)
        sfr = float(np.mean(confident_benign[yeval == 1])) if np.any(yeval == 1) else 0.0
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "escalation_rate": esc,
            "silent_failure_rate": sfr,
        }

    # Baseline (Cycle 0)
    cycle_results = [evaluate(tier1, X_t1_test, y_test)]
    t1_lat_p50, t1_lat_p95, t1_lat_p99 = measure_latency(tier1, X_t1_test)
    cycle_results[0]["tier1_latency_p50_ms"] = t1_lat_p50
    cycle_results[0]["tier1_latency_p95_ms"] = t1_lat_p95

    X_t1_train = list(X_t1_seed)
    y_train = list(y_seed)

    pool_size = len(y_pool)
    num_cycles = config.get("feedback_loop", {}).get("num_cycles", 6)
    batch_size = pool_size // num_cycles

    escalated_counts = []

    for cycle in range(1, num_cycles + 1):
        start = (cycle - 1) * batch_size
        end = start + batch_size if cycle < num_cycles else pool_size

        X_batch_t1 = X_t1_pool[start:end]
        X_batch_t2 = X_t2_pool[start:end]

        probs = tier1.predict_proba(X_batch_t1)
        max_p = np.max(probs, axis=1)
        esc_mask = (max_p >= esc_band[0]) & (max_p <= esc_band[1])
        escalated_counts.append(int(np.sum(esc_mask)))

        if np.any(esc_mask):
            t2_preds = tier2.predict(X_batch_t2[esc_mask])
            X_t1_train.extend(X_batch_t1[esc_mask])
            y_train.extend(t2_preds)
            tier1.fit(np.array(X_t1_train), np.array(y_train))

        ev = evaluate(tier1, X_t1_test, y_test)
        t1_lat_p50, t1_lat_p95, _ = measure_latency(tier1, X_t1_test)
        ev["tier1_latency_p50_ms"] = t1_lat_p50
        ev["tier1_latency_p95_ms"] = t1_lat_p95
        cycle_results.append(ev)

    # Measure Tier 2 latency (real model inference only — no sleep/fake overhead)
    t2_times = []
    t2_sample = X_t2[:1]
    for _ in range(20):  # warmup
        tier2.predict(t2_sample)
    for _ in range(100):
        t0 = time.perf_counter()
        tier2.predict(t2_sample)
        t2_times.append((time.perf_counter() - t0) * 1000)
    t2_lat_p50 = float(np.median(t2_times))
    t2_lat_p95 = float(np.percentile(t2_times, 95))

    return {
        "seed": seed,
        "cycle_0_baseline": cycle_results[0],
        f"cycle_{num_cycles}_final": cycle_results[-1],
        "all_cycles": cycle_results,
        "escalated_per_cycle": escalated_counts,
        "tier2_latency_p50_ms": t2_lat_p50,
        "tier2_latency_p95_ms": t2_lat_p95,
    }


def main():
    print("=" * 70)
    print("MAIN PIPELINE EXPERIMENT — Reproducible Multi-Seed Run")
    print("=" * 70)

    config = load_config("configs/main_pipeline.yaml")
    # Load base config too and merge
    base_cfg = load_config("configs/experiment_base.yaml")
    for k, v in base_cfg.items():
        if k not in config:
            config[k] = v

    seeds = config.get("seeds", [42, 71, 137, 256, 512])
    per_seed_results = []

    for seed in seeds:
        result = run_single_seed(seed, config)
        per_seed_results.append(result)
        run_dir = make_run_dir("main_pipeline", seed=seed)
        write_result(run_dir, result, config=config)

    # Aggregate across seeds
    cycle_keys = list(per_seed_results[0]["all_cycles"][0].keys())
    num_cycles = len(per_seed_results[0]["all_cycles"])
    aggregated_cycles = []
    for c in range(num_cycles):
        cycle_agg = {"cycle": c}
        for key in cycle_keys:
            if isinstance(per_seed_results[0]["all_cycles"][c].get(key), float):
                vals = [r["all_cycles"][c][key] for r in per_seed_results]
                m, ci = mean_ci95(vals)
                cycle_agg[f"{key}_mean"] = m
                cycle_agg[f"{key}_ci95"] = ci
        aggregated_cycles.append(cycle_agg)

    summary_metrics = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "num_cycles": num_cycles - 1,
        "aggregated_cycles": aggregated_cycles,
        "baseline_accuracy_mean_pct": aggregated_cycles[0].get("accuracy_mean", float("nan")) * 100,
        "final_accuracy_mean_pct": aggregated_cycles[-1].get("accuracy_mean", float("nan")) * 100,
        "baseline_escalation_mean_pct": aggregated_cycles[0].get("escalation_rate_mean", float("nan")) * 100,
        "final_escalation_mean_pct": aggregated_cycles[-1].get("escalation_rate_mean", float("nan")) * 100,
    }

    summary_dir = make_run_dir("main_pipeline", seed=None)
    write_result(summary_dir, summary_metrics, config=config, extra_meta={"type": "summary"})

    print("\n=== SUMMARY ===")
    for c_agg in aggregated_cycles:
        acc = c_agg.get("accuracy_mean", float("nan")) * 100
        acc_ci = c_agg.get("accuracy_ci95", float("nan")) * 100
        esc = c_agg.get("escalation_rate_mean", float("nan")) * 100
        esc_ci = c_agg.get("escalation_rate_ci95", float("nan")) * 100
        print(f"  Cycle {c_agg['cycle']}: Acc={acc:.2f}% ± {acc_ci:.2f}%, Esc={esc:.2f}% ± {esc_ci:.2f}%")


if __name__ == "__main__":
    main()
