#!/usr/bin/env python3
"""
scripts/run_lofo_standardized_gates.py
──────────────────────────────────────
Evaluates Leave-One-Family-Out (LOFO) with standardized escalation gates on the UNIFORM benchmark:
  - Standardized Full kNN, Mahalanobis, and UNION gates
  - Target budgets: 1%, 2%, 5%, 10%
  - Deployable thresholds from a held-out 50% training calibration split reported side-by-side
    with oracle-matched thresholds on the test set.
  - Per-fold escalated-and-caught counts for Tier 2 under Oracle and LOFO-trained Tier 2.
  - Stratified within-family flow bootstrap confidence intervals (1,000 resamples).
  - Saves results to results/lofo/lofo_standardized_gates_raw.json.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

BENCHMARK_NPZ = "data/processed/benchmark_uniform_dataset.npz"
OUTPUT_JSON = "results/lofo/lofo_standardized_gates_raw.json"

ALL_FAMILIES = [
    "FTP-BruteForce",
    "SSH-BruteForce",
    "DoS-GoldenEye",
    "DoS-Hulk",
    "DoS-Slowloris",
]

BUDGETS = [0.01, 0.02, 0.05, 0.10]
BUDGET_KEYS = ["1%", "2%", "5%", "10%"]
GATE_NAMES = ["Full_kNN", "Mahalanobis", "UNION_Gate"]
CALIBRATION_MODES = ["deployable", "oracle_matched"]
N_BOOT = 1000


def compute_percentile_ci(values: List[float], alpha: float = 0.05, clip_min: float = 0.0, clip_max: float = 100.0) -> Dict:
    arr = np.array(values)
    lo = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    if clip_min is not None:
        lo = max(clip_min, lo)
    if clip_max is not None:
        hi = min(clip_max, hi)
    return {
        "lo": lo,
        "hi": hi,
        "half_width": float((hi - lo) / 2.0),
    }


def main():
    print("=" * 80)
    print(" LOFO STANDARDIZED GATES: DEPLOYABLE CALIBRATION & ORACLE-MATCHED THRESHOLDS")
    print("=" * 80)

    if not os.path.exists(BENCHMARK_NPZ):
        raise FileNotFoundError(f"Missing {BENCHMARK_NPZ}")

    bench = np.load(BENCHMARK_NPZ)
    rng_boot = np.random.RandomState(42)

    results = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "benchmark_dataset": BENCHMARK_NPZ,
        "gates": GATE_NAMES,
        "budgets": BUDGET_KEYS,
        "calibration_modes": CALIBRATION_MODES,
        "folds": {},
        "macro_summary": {mode: {} for mode in CALIBRATION_MODES},
    }

    # Structure macro accumulator: mode -> gate -> budget -> metric -> list of fold values
    macro_acc = {
        mode: {
            g: {
                b: {
                    "achieved_benign": [],
                    "held_out_coverage": [],
                    "oracle_e2e_miss_rate": [],
                    "oracle_recall": [],
                    "oracle_bacc": [],
                    "oracle_sfr": [],
                    "lofo_e2e_miss_rate": [],
                    "lofo_recall": [],
                    "lofo_bacc": [],
                    "lofo_sfr": [],
                }
                for b in BUDGET_KEYS
            }
            for g in GATE_NAMES
        }
        for mode in CALIBRATION_MODES
    }

    for held_out in ALL_FAMILIES:
        print(f"\nEvaluating Uniform LOFO Fold: Held-out {held_out}...")
        X_tr_9 = bench[f"lofo_{held_out}_X_train"][:, :9]
        X_tr_76 = bench[f"lofo_{held_out}_X_train_76"]
        y_tr = bench[f"lofo_{held_out}_y_train"]

        X_te_9 = bench[f"lofo_{held_out}_X_test"][:, :9]
        X_te_76 = bench[f"lofo_{held_out}_X_test_76"]
        y_te = bench[f"lofo_{held_out}_y_test"]
        fam_te = bench[f"lofo_{held_out}_y_family_test"]

        # Train models
        t1 = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr_9, y_tr)
        t2 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_tr_76, y_tr)

        # Partition training benign into reference (50%) and calibration (50%)
        ben_tr_indices = np.where(y_tr == 0)[0]
        n_ben_tr = len(ben_tr_indices)
        half_ben = n_ben_tr // 2
        ben_ref_indices = ben_tr_indices[:half_ben]
        ben_cal_indices = ben_tr_indices[half_ben:]

        # Standardize 9 features on reference benign train only
        scaler = StandardScaler().fit(X_tr_9[ben_ref_indices])
        Z_ref = scaler.transform(X_tr_9[ben_ref_indices])
        Z_cal = scaler.transform(X_tr_9[ben_cal_indices])
        Z_te = scaler.transform(X_te_9)

        # Base predictions
        p1_prob_cal = t1.predict_proba(X_tr_9[ben_cal_indices])[:, 1]
        p1_prob_te = t1.predict_proba(X_te_9)[:, 1]
        p1_pred_te = (p1_prob_te >= 0.50).astype(int)
        p2_pred_te = t2.predict(X_te_76)

        # Density models fit on Z_ref
        nn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(Z_ref[:3000])
        knn_cal = nn.kneighbors(Z_cal, n_neighbors=5)[0].mean(axis=1)
        knn_te = nn.kneighbors(Z_te, n_neighbors=5)[0].mean(axis=1)

        cov = np.cov(Z_ref, rowvar=False) + 1e-4 * np.eye(Z_ref.shape[1])
        inv_cov = np.linalg.pinv(cov)
        diff_cal = Z_cal - np.mean(Z_ref, axis=0)
        diff_te = Z_te - np.mean(Z_ref, axis=0)
        maha_cal = np.sqrt(np.sum(diff_cal @ inv_cov * diff_cal, axis=1))
        maha_te = np.sqrt(np.sum(diff_te @ inv_cov * diff_te, axis=1))

        rf_unc_cal = 1.0 - np.abs(p1_prob_cal - 0.5) * 2.0
        rf_unc_te = 1.0 - np.abs(p1_prob_te - 0.5) * 2.0

        scores_cal = {
            "Full_kNN": knn_cal,
            "Mahalanobis": maha_cal,
        }
        scores_te = {
            "Full_kNN": knn_te,
            "Mahalanobis": maha_te,
        }

        ben_te_m = (y_te == 0)
        atk_te_m = (y_te == 1)
        held_m = (fam_te == held_out)
        n_held = int(np.sum(held_m))

        # Set up strata for within-family flow bootstrap
        strata = {"Benign": np.where(ben_te_m)[0]}
        for f in ALL_FAMILIES:
            strata[f] = np.where(fam_te == f)[0]

        boot_indices = [
            np.concatenate([rng_boot.choice(idx_list, size=len(idx_list), replace=True) for idx_list in strata.values()])
            for _ in range(N_BOOT)
        ]

        fold_dict = {"deployable": {}, "oracle_matched": {}}

        for mode in CALIBRATION_MODES:
            for g_name in GATE_NAMES:
                fold_dict[mode][g_name] = {}
                for b_val, b_key in zip(BUDGETS, BUDGET_KEYS):
                    if mode == "deployable":
                        # Threshold calibrated on held-out 50% training calibration split
                        if g_name in scores_cal:
                            th = float(np.percentile(scores_cal[g_name], 100.0 * (1.0 - b_val)))
                            esc_mask = scores_te[g_name] >= th
                        else:
                            # UNION Gate: RF uncertainty + kNN on calibration set
                            sub_b = b_val / 2.0
                            rf_th = float(np.percentile(rf_unc_cal, 100.0 * (1.0 - sub_b)))
                            knn_th = float(np.percentile(knn_cal, 100.0 * (1.0 - sub_b)))
                            esc_mask = (rf_unc_te >= rf_th) | (knn_te >= knn_th)
                    else:
                        # Oracle-matched threshold directly on test benign
                        if g_name in scores_te:
                            th = float(np.percentile(scores_te[g_name][ben_te_m], 100.0 * (1.0 - b_val)))
                            esc_mask = scores_te[g_name] >= th
                        else:
                            sub_b = b_val / 2.0
                            rf_th = float(np.percentile(rf_unc_te[ben_te_m], 100.0 * (1.0 - sub_b)))
                            knn_th = float(np.percentile(knn_te[ben_te_m], 100.0 * (1.0 - sub_b)))
                            esc_mask = (rf_unc_te >= rf_th) | (knn_te >= knn_th)

                    # Point Estimates
                    achieved_ben = float(np.mean(esc_mask[ben_te_m]) * 100.0)
                    n_held_esc = int(np.sum(esc_mask & held_m))
                    held_cov = float(n_held_esc / max(1, n_held) * 100.0)

                    # Tier 1 direct catch on held-out (flows predicted attack that are NOT escalated)
                    t1_direct_caught = int(np.sum((p1_pred_te == 1) & (~esc_mask) & held_m))

                    # Oracle Tier 2
                    y_sys_orc = p1_pred_te.copy()
                    y_sys_orc[esc_mask] = y_te[esc_mask]
                    orc_held_caught = int(np.sum((y_sys_orc == 1) & held_m))
                    orc_held_miss = n_held - orc_held_caught
                    orc_held_miss_rate = float(orc_held_miss / max(1, n_held) * 100.0)
                    orc_rec = float(orc_held_caught / max(1, n_held) * 100.0)
                    orc_bacc = float(balanced_accuracy_score(y_te, y_sys_orc) * 100.0)
                    orc_silent = (p1_pred_te == 0) & (~esc_mask) & atk_te_m
                    orc_sfr = float(np.sum(orc_silent) / max(1, np.sum(atk_te_m)) * 100.0)

                    # LOFO-trained Tier 2
                    y_sys_lofo = p1_pred_te.copy()
                    y_sys_lofo[esc_mask] = p2_pred_te[esc_mask]
                    lofo_held_caught = int(np.sum((y_sys_lofo == 1) & held_m))
                    lofo_held_miss = n_held - lofo_held_caught
                    lofo_held_miss_rate = float(lofo_held_miss / max(1, n_held) * 100.0)
                    lofo_rec = float(lofo_held_caught / max(1, n_held) * 100.0)
                    lofo_bacc = float(balanced_accuracy_score(y_te, y_sys_lofo) * 100.0)
                    lofo_silent = (p1_pred_te == 0) & (~esc_mask) & atk_te_m
                    lofo_sfr = float(np.sum(lofo_silent) / max(1, np.sum(atk_te_m)) * 100.0)

                    # Tier 2 specific counts
                    t2_caught_among_esc = int(np.sum((p2_pred_te == 1) & esc_mask & held_m))
                    t2_allowed_among_esc = n_held_esc - t2_caught_among_esc

                    # Stratified bootstrap CIs
                    b_cov, b_orc_miss, b_orc_rec, b_orc_bacc, b_orc_sfr = [], [], [], [], []
                    b_lofo_miss, b_lofo_rec, b_lofo_bacc, b_lofo_sfr = [], [], [], []

                    for s_idx in boot_indices:
                        b_fam = fam_te[s_idx]
                        b_y = y_te[s_idx]
                        b_hm = (b_fam == held_out)
                        b_atk = (b_y == 1)
                        b_esc = esc_mask[s_idx]
                        n_b_h = np.sum(b_hm)

                        b_cov.append(float(np.sum(b_esc & b_hm) / max(1, n_b_h) * 100.0))

                        b_orc = y_sys_orc[s_idx]
                        orc_c = np.sum((b_orc == 1) & b_hm)
                        b_orc_rec.append(float(orc_c / max(1, n_b_h) * 100.0))
                        b_orc_miss.append(float((n_b_h - orc_c) / max(1, n_b_h) * 100.0))
                        b_orc_bacc.append(float(balanced_accuracy_score(b_y, b_orc) * 100.0))
                        b_orc_sfr.append(float(np.sum((p1_pred_te[s_idx] == 0) & (~b_esc) & b_atk) / max(1, np.sum(b_atk)) * 100.0))

                        b_lofo = y_sys_lofo[s_idx]
                        lofo_c = np.sum((b_lofo == 1) & b_hm)
                        b_lofo_rec.append(float(lofo_c / max(1, n_b_h) * 100.0))
                        b_lofo_miss.append(float((n_b_h - lofo_c) / max(1, n_b_h) * 100.0))
                        b_lofo_bacc.append(float(balanced_accuracy_score(b_y, b_lofo) * 100.0))
                        b_lofo_sfr.append(float(np.sum((p1_pred_te[s_idx] == 0) & (~b_esc) & b_atk) / max(1, np.sum(b_atk)) * 100.0))

                    fold_dict[mode][g_name][b_key] = {
                        "calibration_mode": mode,
                        "achieved_benign": achieved_ben,
                        "held_out_n_flows": n_held,
                        "t1_directly_caught": t1_direct_caught,
                        "held_out_n_escalated": n_held_esc,
                        "held_out_coverage_point": held_cov,
                        "held_out_coverage_ci": compute_percentile_ci(b_cov),
                        "oracle": {
                            "t2_caught_among_esc": n_held_esc,
                            "e2e_miss_count": orc_held_miss,
                            "e2e_miss_rate_point": orc_held_miss_rate,
                            "e2e_miss_rate_ci": compute_percentile_ci(b_orc_miss),
                            "system_recall_point": orc_rec,
                            "system_recall_ci": compute_percentile_ci(b_orc_rec),
                            "system_bacc_point": orc_bacc,
                            "system_bacc_ci": compute_percentile_ci(b_orc_bacc),
                            "system_sfr_point": orc_sfr,
                            "system_sfr_ci": compute_percentile_ci(b_orc_sfr),
                        },
                        "lofo_trained_t2": {
                            "t2_caught_among_esc": t2_caught_among_esc,
                            "t2_allowed_among_esc": t2_allowed_among_esc,
                            "total_system_caught": lofo_held_caught,
                            "e2e_miss_count": lofo_held_miss,
                            "e2e_miss_rate_point": lofo_held_miss_rate,
                            "e2e_miss_rate_ci": compute_percentile_ci(b_lofo_miss),
                            "system_recall_point": lofo_rec,
                            "system_recall_ci": compute_percentile_ci(b_lofo_rec),
                            "system_bacc_point": lofo_bacc,
                            "system_bacc_ci": compute_percentile_ci(b_lofo_bacc),
                            "system_sfr_point": lofo_sfr,
                            "system_sfr_ci": compute_percentile_ci(b_lofo_sfr),
                        },
                    }

                    # Accumulate for macro
                    macro_acc[mode][g_name][b_key]["achieved_benign"].append(achieved_ben)
                    macro_acc[mode][g_name][b_key]["held_out_coverage"].append(held_cov)
                    macro_acc[mode][g_name][b_key]["oracle_e2e_miss_rate"].append(orc_held_miss_rate)
                    macro_acc[mode][g_name][b_key]["oracle_recall"].append(orc_rec)
                    macro_acc[mode][g_name][b_key]["oracle_bacc"].append(orc_bacc)
                    macro_acc[mode][g_name][b_key]["oracle_sfr"].append(orc_sfr)
                    macro_acc[mode][g_name][b_key]["lofo_e2e_miss_rate"].append(lofo_held_miss_rate)
                    macro_acc[mode][g_name][b_key]["lofo_recall"].append(lofo_rec)
                    macro_acc[mode][g_name][b_key]["lofo_bacc"].append(lofo_bacc)
                    macro_acc[mode][g_name][b_key]["lofo_sfr"].append(lofo_sfr)

        results["folds"][held_out] = fold_dict

    # Compute Macro Summaries across folds
    for mode in CALIBRATION_MODES:
        for g_name in GATE_NAMES:
            results["macro_summary"][mode][g_name] = {}
            for b_key in BUDGET_KEYS:
                results["macro_summary"][mode][g_name][b_key] = {}
                for metric, vals in macro_acc[mode][g_name][b_key].items():
                    pt = float(np.mean(vals))
                    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                    ci95_t = float(2.776445 * se)
                    results["macro_summary"][mode][g_name][b_key][metric] = {
                        "point_estimate": pt,
                        "se": se,
                        "df": len(vals) - 1,
                        "t_crit": 2.776445,
                        "ci95_t": ci95_t,
                        "lo": max(0.0, pt - ci95_t),
                        "hi": min(100.0, pt + ci95_t),
                        "per_fold_values": vals,
                    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Standardized Gates] Successfully saved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
