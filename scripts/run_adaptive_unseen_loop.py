"""
scripts/run_adaptive_unseen_loop.py
───────────────────────────────────
Implements Task 4: Re-running Algorithm 2 (Adaptive Feedback Loop) on unseen families.
Evaluates:
  1. Shuffled stream containing unseen attack families (SSH, GoldenEye, Hulk, Slowloris)
     alongside previously learned traffic (FTP-BruteForce + Benign).
  2. Strong Tier 1: Tuned Random Forest (n_estimators=50, max_depth=8).
  3. Condition A: Oracle Tier 2 labels (upper bound) - escalated flows receive ground-truth labels.
  4. Condition B: Trained Tier 2 (realistic) - escalated flows receive predictions from Tier 2 RF (76 features).
  5. Retraining Gate: empty batch = skip, FPR drift tolerance = 1.5%, historical regression tolerance = 2.0%.
  6. Per cycle metrics:
     - Held-out family recall (%)
     - Stream escalation rate (%)
     - Benign FPR (%)
     - Accuracy on previously learned families (FTP-BruteForce + Benign) (%)
  7. 5 seeds [42, 71, 137, 256, 512] with bootstrap 95% confidence intervals.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.production_loop import (
    CapacityController,
    RetrainingGate,
    apply_poisoning_bound,
)

BENCHMARK_NPZ = "data/processed/benchmark_dataset.npz"
SEEDS = [42, 71, 137, 256, 512]
UNSEEN_FAMILIES = ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]
KNOWN_FAMILIES = ["FTP-BruteForce"]


def bootstrap_ci95(values: List[float], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """Compute bootstrap mean and 95% confidence interval half-width."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    low, high = np.percentile(boot_means, [2.5, 97.5])
    mean_val = float(np.mean(arr))
    ci_half = float((high - low) / 2.0)
    return mean_val, ci_half


def run_adaptive_loop_simulation(oracle_tier2: bool = True, seeds: List[int] = SEEDS) -> Dict[str, Any]:
    cond_name = "Oracle Tier 2 (Upper Bound)" if oracle_tier2 else "Trained Tier 2 (Realistic)"
    print(f"\n[Algorithm 2] Running simulation under: {cond_name}")

    bench = np.load(BENCHMARK_NPZ)
    X_init_9 = bench["ftp_only_X_train"]
    X_init_76 = bench["ftp_only_X_train_76"]
    y_init = bench["ftp_only_y_train"]

    X_test_9 = bench["ftp_only_X_test"]
    X_test_76 = bench["ftp_only_X_test_76"]
    y_test = bench["ftp_only_y_test"]
    fam_test = bench["ftp_only_y_family_test"]

    # Historical subsets for gate regression check
    ftp_mask_test = (fam_test == "FTP-BruteForce")
    benign_mask_test = (fam_test == "Benign")
    unseen_mask_test = np.isin(fam_test, UNSEEN_FAMILIES)

    seed_runs = []

    for seed in seeds:
        print(f"  Starting Seed {seed}...")
        rng = np.random.RandomState(seed)

        # Baseline Tier 1: Strong Random Forest on 9 features
        t1_base = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=seed, n_jobs=-1)
        t1_base.fit(X_init_9, y_init)

        # Realistic Tier 2: Random Forest on 76 features
        t2_realistic = None
        if not oracle_tier2:
            t2_realistic = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=seed, n_jobs=-1)
            t2_realistic.fit(X_init_76, y_init)

        # Controller and Gate
        controller = CapacityController(
            capacity_req_per_sec=40.0,
            max_queue_depth=100,
            overflow_policy="default-to-tier1",
            initial_band=(0.30, 0.70),
        )
        gate = RetrainingGate(max_fpr_drift=0.015, max_acc_drop=0.02)
        gate.register_initial_model(t1_base)

        # Separate held-out validation set for the gate (800 flows)
        val_idx = rng.choice(len(X_test_9), size=800, replace=False)
        X_val_9 = X_test_9[val_idx]
        y_val = y_test[val_idx]
        fam_val = fam_test[val_idx]

        # Construct historical verification subsets
        hist_subsets = {}
        for kf in KNOWN_FAMILIES:
            km = (fam_val == kf)
            if np.sum(km) > 10:
                hist_subsets[kf] = (X_val_9[km], y_val[km])
            else:
                km_full = (fam_test == kf)
                hist_subsets[kf] = (X_test_9[km_full][:100], y_test[km_full][:100])

        # Streaming pool (excluding validation set)
        stream_mask = np.ones(len(X_test_9), dtype=bool)
        stream_mask[val_idx] = False

        X_stream_9 = X_test_9[stream_mask]
        X_stream_76 = X_test_76[stream_mask]
        y_stream = y_test[stream_mask]
        fam_stream = fam_test[stream_mask]

        # Shuffled stream
        perm = rng.permutation(len(X_stream_9))
        X_stream_9 = X_stream_9[perm]
        X_stream_76 = X_stream_76[perm]
        y_stream = y_stream[perm]
        fam_stream = fam_stream[perm]

        cycle_size = len(X_stream_9) // 5
        cycle_records = []

        # Current training pool
        X_tr_curr_9 = X_init_9.copy()
        y_tr_curr = y_init.copy()
        w_tr_curr = np.ones(len(y_init))

        # ── Cycle 0: Baseline Evaluation ──────────────────────────────────────
        active_model = gate.model_versions[gate.active_version]
        p_eval = active_model.predict_proba(X_test_9)[:, 1]
        preds_eval = (p_eval >= 0.5).astype(int)

        band_0 = controller.get_current_band()
        esc_mask_0 = (p_eval >= band_0[0]) & (p_eval <= band_0[1])

        # Held-out recall (on unseen families test set)
        unseen_rec_0 = float(np.mean(preds_eval[unseen_mask_test] == 1) * 100.0)
        # Stream escalation rate
        esc_rate_0 = float(np.mean(esc_mask_0) * 100.0)
        # Benign FPR
        benign_fpr_0 = float(np.mean(preds_eval[benign_mask_test] == 1) * 100.0)
        # Accuracy on previously learned families (FTP + Benign)
        prev_learned_mask = benign_mask_test | ftp_mask_test
        prev_acc_0 = float(accuracy_score(y_test[prev_learned_mask], preds_eval[prev_learned_mask]) * 100.0)

        cycle_records.append({
            "cycle": 0,
            "version": gate.active_version,
            "held_out_recall": unseen_rec_0,
            "escalation_rate": esc_rate_0,
            "benign_fpr": benign_fpr_0,
            "prev_learned_acc": prev_acc_0,
            "gate_status": "DEPLOYED",
            "gate_reason": "Baseline model initialization",
        })

        # ── Cycles 1 to 5: Sequential Streaming Adaptation ───────────────────
        for c in range(1, 6):
            c_start = (c - 1) * cycle_size
            c_end = c_start + cycle_size if c < 5 else len(X_stream_9)

            X_batch_9 = X_stream_9[c_start:c_end]
            X_batch_76 = X_stream_76[c_start:c_end]
            y_batch = y_stream[c_start:c_end]
            fam_batch = fam_stream[c_start:c_end]

            active_model = gate.model_versions[gate.active_version]
            probs_batch = active_model.predict_proba(X_batch_9)[:, 1]
            preds_batch = (probs_batch >= 0.5).astype(int)

            # Controller routing
            escalated_indices = []
            for i, p in enumerate(probs_batch):
                is_esc, _, _ = controller.route_escalation(p, preds_batch[i])
                if is_esc:
                    escalated_indices.append(i)

            controller.step_controller(processed_requests=len(X_batch_9), elapsed_sec=2.0)

            cand_metrics = {}
            if len(escalated_indices) > 0:
                esc_idx_arr = np.array(escalated_indices)
                new_X_9 = X_batch_9[esc_idx_arr]

                # Assign labels from Tier 2
                if oracle_tier2:
                    new_y = y_batch[esc_idx_arr]
                else:
                    new_y = t2_realistic.predict(X_batch_76[esc_idx_arr])

                conf_scores = np.abs(probs_batch[esc_idx_arr] - 0.5) * 2.0

                X_tr_curr_9, y_tr_curr, w_tr_curr = apply_poisoning_bound(
                    X_tr_curr_9,
                    y_tr_curr,
                    w_tr_curr,
                    new_X_9,
                    new_y,
                    max_influence_pct=0.20,
                    confidence_scores=conf_scores,
                )

                cand_version = f"v1.{c}"
                cand_model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=seed, n_jobs=-1)
                cand_model.fit(X_tr_curr_9, y_tr_curr, sample_weight=w_tr_curr)

                passed, gate_reason, cand_metrics = gate.evaluate_gate(
                    candidate_model=cand_model,
                    val_X=X_val_9,
                    val_y=y_val,
                    historical_subsets=hist_subsets,
                    candidate_version_name=cand_version,
                )
                gate_status = "PASSED" if passed else "REJECTED_ROLLBACK"
            else:
                passed, gate_reason, cand_metrics = gate.evaluate_gate(
                    candidate_model=None,
                    val_X=X_val_9,
                    val_y=y_val,
                    historical_subsets=hist_subsets,
                    candidate_version_name=f"v1.{c}",
                )
                gate_status = "SKIPPED_EMPTY_BATCH"

            # Evaluate active deployed model
            active_deployed = gate.model_versions[gate.active_version]
            p_test = active_deployed.predict_proba(X_test_9)[:, 1]
            preds_test = (p_test >= 0.5).astype(int)

            band_c = controller.get_current_band()
            esc_mask_c = (p_test >= band_c[0]) & (p_test <= band_c[1])

            unseen_rec_c = float(np.mean(preds_test[unseen_mask_test] == 1) * 100.0)
            esc_rate_c = float(np.mean(esc_mask_c) * 100.0)
            benign_fpr_c = float(np.mean(preds_test[benign_mask_test] == 1) * 100.0)
            prev_acc_c = float(accuracy_score(y_test[prev_learned_mask], preds_test[prev_learned_mask]) * 100.0)

            cycle_records.append({
                "cycle": c,
                "version": gate.active_version,
                "held_out_recall": unseen_rec_c,
                "escalation_rate": esc_rate_c,
                "benign_fpr": benign_fpr_c,
                "prev_learned_acc": prev_acc_c,
                "gate_status": gate_status,
                "gate_reason": gate_reason,
            })
            print(f"    Cycle {c}: Held-out Rec={unseen_rec_c:.2f}%, Esc={esc_rate_c:.2f}%, FPR={benign_fpr_c:.2f}%, PrevAcc={prev_acc_c:.2f}%, Active={gate.active_version}")

        seed_runs.append(cycle_records)

    # Aggregate across seeds with bootstrap 95% CI
    num_cycles = len(seed_runs[0])
    agg_cycles = []
    for c in range(num_cycles):
        cm = {"cycle": c}
        for metric_k in ["held_out_recall", "escalation_rate", "benign_fpr", "prev_learned_acc"]:
            vals = [run[c][metric_k] for run in seed_runs]
            m, ci = bootstrap_ci95(vals, n_boot=2000, seed=42 + c)
            cm[f"{metric_k}_mean"] = m
            cm[f"{metric_k}_ci95"] = ci

        # Representative status
        cm["active_version"] = seed_runs[0][c]["version"]
        cm["gate_status"] = seed_runs[0][c]["gate_status"]
        agg_cycles.append(cm)

    return {
        "condition": cond_name,
        "oracle_tier2": oracle_tier2,
        "seeds": seeds,
        "aggregated_cycles": agg_cycles,
        "seed_runs": seed_runs,
    }


def main():
    print("=" * 70)
    print("TASK 4: ADAPTIVE FEEDBACK LOOP BENCHMARK ON UNSEEN FAMILIES")
    print("Comparing Condition A (Oracle Tier 2) vs Condition B (Trained Tier 2)")
    print("=" * 70)

    res_oracle = run_adaptive_loop_simulation(oracle_tier2=True, seeds=SEEDS)
    res_trained = run_adaptive_loop_simulation(oracle_tier2=False, seeds=SEEDS)

    out_dir = Path("results/adaptive_loop")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "unseen_adaptation_summary.json"

    combined = {
        "oracle_tier2_upper_bound": res_oracle,
        "trained_tier2_realistic": res_trained,
        "seeds": SEEDS,
    }

    with open(summary_file, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nSaved Adaptive Feedback Loop results to: {summary_file}")


if __name__ == "__main__":
    main()
