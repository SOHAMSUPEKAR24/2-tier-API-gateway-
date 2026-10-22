#!/usr/bin/env python3
"""
scripts/run_standardized_algorithm2_sweep.py
────────────────────────────────────────────
Re-runs Algorithm 2 (Adaptive Retraining Loop) with standardized gates:
  - Standardized kNN gate (k=5, Euclidean, StandardScaler fit on clean benign train)
  - Gate reference set, scaler, and threshold are kept STATIC across cycles to prevent
    contamination and drift from unverified streaming flows.
  - 4 Evaluation configurations:
    1. L = infinity (unlimited label budget), with rehearsal
    2. L = infinity (unlimited label budget), without rehearsal
    3. L = 50 (finite label budget), with rehearsal
    4. L = 50 (finite label budget), without rehearsal
  - 5 seeds: [42, 71, 137, 256, 512]
  - Per cycle (1 to 5) and per family (SSH, GoldenEye, Hulk, Slowloris, FTP, Benign):
    * Standalone Tier 1 Recall (%)
    * System Recall under Oracle (%)
    * System Recall under Trained Tier 2 (%)
    * Escalation Coverage (%)
    * Benign FPR (%)
  - Outputs to results/adaptive_loop/standardized_algorithm2_runs.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.production_loop import RetrainingGate, apply_poisoning_bound

BENCHMARK_NPZ = "data/processed/benchmark_dataset.npz"
OUT_JSON = "results/adaptive_loop/standardized_algorithm2_runs.json"
SEEDS = [42, 71, 137, 256, 512]
NOVEL_FAMILIES = ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]
ALL_FAMILIES = ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris", "FTP-BruteForce"]


def bootstrap_ci95(values: List[float], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(np.mean(arr)), float((high - low) / 2.0)


def run_algorithm2_sweep():
    print("=" * 70)
    print("RUNNING STANDARDIZED ALGORITHM 2 ADAPTIVE RETRAINING SWEEP")
    print("=" * 70)

    if not os.path.exists(BENCHMARK_NPZ):
        raise FileNotFoundError(f"Missing benchmark dataset: {BENCHMARK_NPZ}")

    bench = np.load(BENCHMARK_NPZ)
    X_init_9 = bench["ftp_only_X_train"][:, :9]
    y_init = bench["ftp_only_y_train"]
    X_init_76 = bench["ftp_only_X_train_76"]

    X_te_9 = bench["ftp_only_X_test"][:, :9]
    y_te = bench["ftp_only_y_test"]
    y_fam_te = bench["ftp_only_y_family_test"]
    X_te_76 = bench["ftp_only_X_test_76"]

    benign_init = X_init_9[y_init == 0]

    # Standardize density gate on clean benign training data
    scaler = StandardScaler().fit(benign_init)
    benign_init_sc = scaler.transform(benign_init)
    X_te_9_sc = scaler.transform(X_te_9)

    # Reference set: 4000 initial clean benign flows (static, NOT refit each cycle)
    knn_ref = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(benign_init_sc[:4000])
    cal_scores = knn_ref.kneighbors(benign_init_sc[4000:])[0][:, -1]
    tau_5pct = float(np.quantile(cal_scores, 0.95))

    # 10 Chronological blocks on test set for block-bootstrap CIs
    N_BLOCKS = 10
    block_size = len(y_te) // N_BLOCKS
    test_blocks = [
        np.arange(i * block_size, (i + 1) * block_size if i < N_BLOCKS - 1 else len(y_te))
        for i in range(N_BLOCKS)
    ]

    sweep_configs = [
        {
            "name": "oracle_L_inf_with_rehearsal",
            "display_name": "Oracle Tier 2, L=inf, With Rehearsal",
            "label_source": "oracle",
            "is_oracle": True,
            "L_max": 100000,
            "with_rehearsal": True,
            "gate_mode": "on",
            "gate_description": "Gate ON (Rollback on regression)",
        },
        {
            "name": "oracle_L_inf_no_rehearsal_rollback",
            "display_name": "Oracle Tier 2, L=inf, No Rehearsal (Rollback)",
            "label_source": "oracle",
            "is_oracle": True,
            "L_max": 100000,
            "with_rehearsal": False,
            "gate_mode": "on",
            "gate_description": "Gate ON (Rollback to baseline on regression)",
        },
        {
            "name": "oracle_L_inf_no_rehearsal_gate_off",
            "display_name": "Oracle Tier 2, L=inf, No Rehearsal (Gate OFF: True Forgetting)",
            "label_source": "oracle",
            "is_oracle": True,
            "L_max": 100000,
            "with_rehearsal": False,
            "gate_mode": "off",
            "gate_description": "Gate OFF (Unconditional deployment, measuring true forgetting)",
        },
        {
            "name": "oracle_L_50_with_rehearsal",
            "display_name": "Oracle Tier 2, L=50, With Rehearsal",
            "label_source": "oracle",
            "is_oracle": True,
            "L_max": 50,
            "with_rehearsal": True,
            "gate_mode": "on",
            "gate_description": "Gate ON (Rollback on regression)",
        },
        {
            "name": "oracle_L_50_no_rehearsal_rollback",
            "display_name": "Oracle Tier 2, L=50, No Rehearsal (Rollback)",
            "label_source": "oracle",
            "is_oracle": True,
            "L_max": 50,
            "with_rehearsal": False,
            "gate_mode": "on",
            "gate_description": "Gate ON (Rollback to baseline on regression)",
        },
        {
            "name": "oracle_L_50_no_rehearsal_gate_off",
            "display_name": "Oracle Tier 2, L=50, No Rehearsal (Gate OFF: True Forgetting)",
            "label_source": "oracle",
            "is_oracle": True,
            "L_max": 50,
            "with_rehearsal": False,
            "gate_mode": "off",
            "gate_description": "Gate OFF (Unconditional deployment, measuring true forgetting)",
        },
        {
            "name": "trained_t2_L_inf_with_rehearsal",
            "display_name": "Trained Tier 2 (Pseudo-Labels), L=inf, With Rehearsal",
            "label_source": "trained_t2",
            "is_oracle": False,
            "L_max": 100000,
            "with_rehearsal": True,
            "gate_mode": "on",
            "gate_description": "Gate ON (Rollback on regression)",
        },
    ]

    all_config_results = {}

    for cfg in sweep_configs:
        cname = cfg["name"]
        l_max = cfg["L_max"]
        with_reh = cfg["with_rehearsal"]
        lbl_src = cfg["label_source"]
        g_mode = cfg["gate_mode"]
        print(f"\n--- Running Config: {cfg['display_name']} ---")

        seed_cycle_records = []
        seed_cycle_block_records = []

        for seed in SEEDS:
            rng = np.random.RandomState(seed)

            # Initial Tier 1 (40 trees, depth 6)
            t1 = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed, n_jobs=-1).fit(
                X_init_9, y_init
            )

            # Realistic Tier 2 (50 trees, depth 10 on 76 features)
            t2 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=seed, n_jobs=-1).fit(
                X_init_76, y_init
            )
            preds_t2_test = t2.predict(X_te_76)

            gate = RetrainingGate(max_fpr_drift=0.015, max_acc_drop=0.02)
            gate.register_initial_model(t1)

            # Validation split (800 flows)
            val_idx = rng.choice(len(X_te_9), size=800, replace=False)
            X_val_9 = X_te_9[val_idx]
            y_val = y_te[val_idx]
            fam_val = y_fam_te[val_idx]

            hist_subsets = {}
            for ef in ["FTP-BruteForce"]:
                km = fam_val == ef
                if np.sum(km) > 10:
                    hist_subsets[ef] = (X_val_9[km], y_val[km])
                else:
                    km_f = y_fam_te == ef
                    hist_subsets[ef] = (X_te_9[km_f][:100], y_te[km_f][:100])

            # Streaming split
            stream_mask = np.ones(len(X_te_9), dtype=bool)
            stream_mask[val_idx] = False
            X_st_9 = X_te_9[stream_mask]
            X_st_76 = X_te_76[stream_mask]
            y_st = y_te[stream_mask]
            fam_st = y_fam_te[stream_mask]

            # Shuffled stream
            perm = rng.permutation(len(X_st_9))
            X_st_9 = X_st_9[perm]
            X_st_76 = X_st_76[perm]
            y_st = y_st[perm]
            fam_st = fam_st[perm]

            cycle_size = len(X_st_9) // 5

            X_tr_curr = X_init_9.copy() if with_reh else np.empty((0, 9))
            y_tr_curr = y_init.copy() if with_reh else np.empty(0, dtype=int)
            w_tr_curr = np.ones(len(y_init)) if with_reh else np.empty(0)

            c_logs = []
            c_block_logs = []

            # Pre-adaptation evaluation (Cycle 0)
            active_m0 = gate.model_versions[gate.active_version]
            p0 = active_m0.predict(X_te_9)
            scores_te = knn_ref.kneighbors(X_te_9_sc)[0][:, -1]
            esc_te = scores_te >= tau_5pct

            c0_entry = {"cycle": 0, "active_version": gate.active_version}
            for fam in ALL_FAMILIES:
                fm = y_fam_te == fam
                c0_entry[f"rec_t1_{fam}"] = float(np.mean(p0[fm] == 1) * 100.0) if np.sum(fm) > 0 else 0.0
                c0_entry[f"rec_sys_oracle_{fam}"] = (
                    float(np.mean((p0[fm] == 1) | esc_te[fm]) * 100.0) if np.sum(fm) > 0 else 0.0
                )
                c0_entry[f"rec_sys_trained_{fam}"] = (
                    float(
                        np.mean(((p0[fm] == 1) & (~esc_te[fm])) | (esc_te[fm] & (preds_t2_test[fm] == 1)))
                        * 100.0
                    )
                    if np.sum(fm) > 0
                    else 0.0
                )
                c0_entry[f"esc_cov_{fam}"] = float(np.mean(esc_te[fm]) * 100.0) if np.sum(fm) > 0 else 0.0

            c0_entry["benign_fpr"] = float(np.mean(p0[y_fam_te == "Benign"] == 1) * 100.0)
            c_logs.append(c0_entry)

            # Block-level evaluation for Cycle 0
            c0_blocks = []
            for b_idx in test_blocks:
                b_entry = {}
                for fam in ALL_FAMILIES:
                    fm_b = (y_fam_te[b_idx] == fam)
                    b_entry[f"rec_t1_{fam}"] = float(np.mean(p0[b_idx][fm_b] == 1) * 100.0) if np.sum(fm_b) > 0 else np.nan
                b_entry["benign_fpr"] = float(np.mean(p0[b_idx][y_fam_te[b_idx] == "Benign"] == 1) * 100.0)
                c0_blocks.append(b_entry)
            c_block_logs.append(c0_blocks)

            # Cycles 1 to 5
            for c in range(1, 6):
                c_start = (c - 1) * cycle_size
                c_end = c_start + cycle_size if c < 5 else len(X_st_9)
                X_b = X_st_9[c_start:c_end]
                X_b_76 = X_st_76[c_start:c_end]
                y_b = y_st[c_start:c_end]
                fam_b = fam_st[c_start:c_end]

                # Density scoring with static standardized gate
                X_b_sc = scaler.transform(X_b)
                d_scores = knn_ref.kneighbors(X_b_sc)[0][:, -1]
                esc_mask_b = d_scores >= tau_5pct
                esc_idx = np.where(esc_mask_b)[0]

                if len(esc_idx) > 0:
                    if len(esc_idx) > l_max:
                        top_idx = esc_idx[np.argsort(d_scores[esc_idx])[-l_max:]]
                        labeled_idx = top_idx
                    else:
                        labeled_idx = esc_idx

                    new_X = X_b[labeled_idx]

                    # Label assignment: Oracle vs Trained Tier 2 Pseudo-Labels
                    if lbl_src == "trained_t2":
                        new_y = t2.predict(X_b_76[labeled_idx])
                    else:
                        new_y = y_b[labeled_idx]

                    if with_reh:
                        X_tr_curr, y_tr_curr, w_tr_curr = apply_poisoning_bound(
                            X_tr_curr, y_tr_curr, w_tr_curr, new_X, new_y, max_influence_pct=0.20
                        )
                    else:
                        X_tr_curr = new_X.copy()
                        y_tr_curr = new_y.copy()
                        w_tr_curr = np.ones(len(new_y))

                    cand = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed, n_jobs=-1)
                    cand.fit(X_tr_curr, y_tr_curr, sample_weight=w_tr_curr)

                    if g_mode == "off":
                        # Gate OFF: unconditional deployment to measure true catastrophic forgetting
                        gate.model_versions[f"v1.{c}"] = cand
                        gate.active_version = f"v1.{c}"
                    else:
                        # Gate ON: check against baseline historical performance, rollback on regression
                        gate.evaluate_gate(
                            candidate_model=cand,
                            val_X=X_val_9,
                            val_y=y_val,
                            historical_subsets=hist_subsets,
                            candidate_version_name=f"v1.{c}",
                        )

                active_m = gate.model_versions[gate.active_version]
                preds_te = active_m.predict(X_te_9)

                c_entry = {"cycle": c, "active_version": gate.active_version}
                for fam in ALL_FAMILIES:
                    fm = y_fam_te == fam
                    rec_t1 = float(np.mean(preds_te[fm] == 1) * 100.0) if np.sum(fm) > 0 else 0.0
                    rec_sys_orc = (
                        float(np.mean((preds_te[fm] == 1) | esc_te[fm]) * 100.0) if np.sum(fm) > 0 else 0.0
                    )
                    rec_sys_trn = (
                        float(
                            np.mean(((preds_te[fm] == 1) & (~esc_te[fm])) | (esc_te[fm] & (preds_t2_test[fm] == 1)))
                            * 100.0
                        )
                        if np.sum(fm) > 0
                        else 0.0
                    )
                    esc_cov = float(np.mean(esc_te[fm]) * 100.0) if np.sum(fm) > 0 else 0.0

                    c_entry[f"rec_t1_{fam}"] = rec_t1
                    c_entry[f"rec_sys_oracle_{fam}"] = rec_sys_orc
                    c_entry[f"rec_sys_trained_{fam}"] = rec_sys_trn
                    c_entry[f"esc_cov_{fam}"] = esc_cov

                c_entry["benign_fpr"] = float(np.mean(preds_te[y_fam_te == "Benign"] == 1) * 100.0)
                c_logs.append(c_entry)

                # Block-level evaluation for Cycle c
                c_blocks = []
                for b_idx in test_blocks:
                    b_entry = {}
                    for fam in ALL_FAMILIES:
                        fm_b = (y_fam_te[b_idx] == fam)
                        b_entry[f"rec_t1_{fam}"] = float(np.mean(preds_te[b_idx][fm_b] == 1) * 100.0) if np.sum(fm_b) > 0 else np.nan
                    b_entry["benign_fpr"] = float(np.mean(preds_te[b_idx][y_fam_te[b_idx] == "Benign"] == 1) * 100.0)
                    c_blocks.append(b_entry)
                c_block_logs.append(c_blocks)

            seed_cycle_records.append(c_logs)
            seed_cycle_block_records.append(c_block_logs)

        # Aggregate across seeds with 10-block bootstrap CIs (1,000 resamples)
        rng_agg = np.random.RandomState(42)
        agg_cycles = []
        for c_idx in range(6):  # cycle 0 to 5
            cycle_summary = {"cycle": c_idx}
            keys = [k for k in seed_cycle_records[0][c_idx].keys() if k not in ["cycle", "active_version"]]

            for k in keys:
                # Overall mean across seeds
                overall_mean = float(np.mean([s_rec[c_idx][k] for s_rec in seed_cycle_records]))
                cycle_summary[f"{k}_mean"] = overall_mean

                # Block-bootstrap over the 10 chronological blocks across seeds
                boot_samples = []
                for _ in range(1000):
                    resampled_blocks = rng_agg.choice(N_BLOCKS, size=N_BLOCKS, replace=True)
                    vals = []
                    for s_idx in range(len(SEEDS)):
                        for b_i in resampled_blocks:
                            v = seed_cycle_block_records[s_idx][c_idx][b_i].get(k, np.nan)
                            if not np.isnan(v):
                                vals.append(v)
                    if vals:
                        boot_samples.append(float(np.mean(vals)))
                if boot_samples:
                    low, high = np.percentile(boot_samples, [2.5, 97.5])
                    cycle_summary[f"{k}_ci95"] = float((high - low) / 2.0)
                else:
                    cycle_summary[f"{k}_ci95"] = 0.0

            agg_cycles.append(cycle_summary)

        all_config_results[cname] = {
            "config": cfg,
            "gate_refit_each_cycle": False,  # Explicitly documented
            "gate_policy": "Static standardized kNN gate fit on clean baseline benign flows",
            "is_oracle": cfg["is_oracle"],
            "label_source": cfg["label_source"],
            "gate_mode": cfg["gate_mode"],
            "cycles": agg_cycles,
        }

    final_payload = {
        "execution_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "gate_status": "Static standardized kNN gate (scaler, reference set, and threshold are NOT refit each cycle to prevent contamination)",
        "configurations": all_config_results,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved Algorithm 2 standardized sweep results to: {OUT_JSON}")


if __name__ == "__main__":
    run_algorithm2_sweep()
