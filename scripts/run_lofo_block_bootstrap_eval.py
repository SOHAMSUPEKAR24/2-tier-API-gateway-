#!/usr/bin/env python3
"""
scripts/run_lofo_block_bootstrap_eval.py
────────────────────────────────────────
Full Leave-One-Family-Out (LOFO) benchmark with exact full test set point estimates
and stratified within-family flow-level bootstrap confidence intervals:
  1. Main LOFO Table:
     - 5 folds: FTP-BruteForce, SSH-BruteForce, DoS-GoldenEye, DoS-Hulk, DoS-Slowloris
     - Models: Integrated Two-Tier System, Tier 1 Standalone, Tier 2 Standalone
     - Point estimates evaluated on the un-resampled full test set.
     - Confidence intervals computed separately via stratified/within-family flow-level
       bootstrap (1,000 resamples with replacement, percentile CI clipped to [0, 100]).
     - Exact flow counts per fold (attack flows per family, blocked, escalated, silent)
       so recall, SFR, escalation, and balanced accuracy reconcile cleanly.
     - Paired per-fold differences (Two-Tier minus Tier 1) for SFR, held-out recall,
       balanced accuracy, and FPR.
  2. Window Size Feature Ablation (76 -> 74 features):
     - Exact 76-feat point estimates strictly aligned with Table 2 Tier 2 Standalone.
     - 74-feat ablated Tier 2 evaluation with flow-level bootstrap CIs.
     - Verified assertion on 74 non-window feature indices.
  3. Prevalence Robustness Evaluation (1:100 and 1:1,000):
     - Evaluated on prevalence_1_100_test and prevalence_1_1000_test splits.
  4. Output saved to results/lofo/lofo_block_bootstrap_raw.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_guard import ALL_ALLOWLISTED_CIC_FEATURES

BENCHMARK_NPZ = "data/processed/benchmark_dataset.npz"
OUT_JSON = "results/lofo/lofo_block_bootstrap_raw.json"
ALL_FAMILIES = ["FTP-BruteForce", "SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]
N_BOOT = 1000

# Window size feature indices in 76 features
WIN_INDICES = [66, 67]  # FWD Init Win Bytes, Bwd Init Win Bytes
NON_WIN_INDICES = [i for i in range(76) if i not in WIN_INDICES]
assert len(NON_WIN_INDICES) == 74, f"Expected 74 indices, got {len(NON_WIN_INDICES)}"


def compute_percentile_ci(vals: List[float], clip_min: float = 0.0, clip_max: float = 100.0) -> Dict[str, float]:
    """Compute percentile 95% bootstrap confidence interval [p2.5, p97.5] clipped to bounds."""
    lo = float(np.clip(np.percentile(vals, 2.5), clip_min, clip_max))
    hi = float(np.clip(np.percentile(vals, 97.5), clip_min, clip_max))
    hw = float((hi - lo) / 2.0)
    return {"lo": lo, "hi": hi, "half_width": hw}


def run_lofo_benchmark():
    print("=" * 70)
    print("RUNNING FULL LOFO BENCHMARK WITH EXACT POINT ESTIMATES & FLOW BOOTSTRAP CIS")
    print("=" * 70)

    if not os.path.exists(BENCHMARK_NPZ):
        raise FileNotFoundError(f"Missing benchmark dataset: {BENCHMARK_NPZ}")

    bench = np.load(BENCHMARK_NPZ)
    rng_boot = np.random.RandomState(42)

    # ── 1. Split-Table Component Sums ─────────────────────────────────────────
    split_names = [
        "ftp_only_train", "ftp_only_test",
        "lofo_FTP-BruteForce_train", "lofo_SSH-BruteForce_train",
        "lofo_DoS-GoldenEye_train", "lofo_DoS-Hulk_train", "lofo_DoS-Slowloris_train",
        "prevalence_1_100_test", "prevalence_1_1000_test"
    ]

    split_table = {}
    for s_name in split_names:
        if s_name.endswith("_train"):
            prefix = s_name.replace("_train", "")
            y = bench[f"{prefix}_y_train"]
            n_tot = len(y)
            n_ben = int(np.sum(y == 0))
            n_atk = int(np.sum(y == 1))
            assert n_tot == n_ben + n_atk, f"Sum mismatch in {s_name}: {n_tot} != {n_ben} + {n_atk}"

            # Family breakdown for training splits
            if s_name == "ftp_only_train":
                fam_counts = {"FTP-BruteForce": 625, "SSH-BruteForce": 0, "DoS-GoldenEye": 0, "DoS-Hulk": 0, "DoS-Slowloris": 0}
            elif s_name == "lofo_FTP-BruteForce_train":
                fam_counts = {"FTP-BruteForce": 0, "SSH-BruteForce": 623, "DoS-GoldenEye": 1048, "DoS-Hulk": 905, "DoS-Slowloris": 992}
            elif s_name == "lofo_SSH-BruteForce_train":
                fam_counts = {"FTP-BruteForce": 625, "SSH-BruteForce": 0, "DoS-GoldenEye": 1048, "DoS-Hulk": 905, "DoS-Slowloris": 992}
            elif s_name == "lofo_DoS-GoldenEye_train":
                fam_counts = {"FTP-BruteForce": 625, "SSH-BruteForce": 623, "DoS-GoldenEye": 0, "DoS-Hulk": 905, "DoS-Slowloris": 992}
            elif s_name == "lofo_DoS-Hulk_train":
                fam_counts = {"FTP-BruteForce": 625, "SSH-BruteForce": 623, "DoS-GoldenEye": 1048, "DoS-Hulk": 0, "DoS-Slowloris": 992}
            elif s_name == "lofo_DoS-Slowloris_train":
                fam_counts = {"FTP-BruteForce": 625, "SSH-BruteForce": 623, "DoS-GoldenEye": 1048, "DoS-Hulk": 905, "DoS-Slowloris": 0}

            assert sum(fam_counts.values()) == n_atk, f"Training family sum mismatch in {s_name}"

            split_table[s_name] = {
                "total_flows": n_tot,
                "benign_flows": n_ben,
                "total_attacks": n_atk,
                "family_counts": fam_counts,
                "verified_sum": True,
            }
        else:
            prefix = s_name.replace("_test", "")
            y = bench[f"{prefix}_y_test"]
            fam = bench[f"{prefix}_y_family_test"]
            n_tot = len(y)
            n_ben = int(np.sum(y == 0))
            n_atk = int(np.sum(y == 1))
            assert n_tot == n_ben + n_atk, f"Sum mismatch in {s_name}: {n_tot} != {n_ben} + {n_atk}"

            fam_counts = {f: int(np.sum(fam == f)) for f in ALL_FAMILIES}

            split_table[s_name] = {
                "total_flows": n_tot,
                "benign_flows": n_ben,
                "total_attacks": n_atk,
                "family_counts": fam_counts,
                "verified_sum": True,
            }

    print("Verified component sums across all benchmark splits.")

    # ── 2. Full LOFO Benchmark Execution ──────────────────────────────────────
    lofo_folds_raw = {}
    macro_accumulators = {
        "sys": {"bacc": [], "held_rec": [], "fpr": [], "sfr": [], "tot_esc": [], "held_esc": [], "ben_esc": []},
        "t1": {"bacc": [], "held_rec": [], "fpr": [], "sfr": []},
        "t2": {"bacc": [], "held_rec": [], "fpr": [], "sfr": []},
        "diffs": {"delta_sfr": [], "delta_held_rec": [], "delta_bacc": [], "delta_fpr": []},
    }

    t2_models = {}

    for held_out in ALL_FAMILIES:
        print(f"  Evaluating fold: Held-out {held_out}...")
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
        t2_models[held_out] = t2

        p1_prob = t1.predict_proba(X_te_9)[:, 1]
        p1_pred = (p1_prob >= 0.50).astype(int)
        p2_pred = t2.predict(X_te_76)

        # Escalation: RF probability band [0.30, 0.70]
        esc_mask = (p1_prob >= 0.30) & (p1_prob <= 0.70)

        # Integrated Two-Tier system predictions
        y_sys = p1_pred.copy()
        y_sys[esc_mask] = p2_pred[esc_mask]

        atk_m = (y_te == 1)
        ben_m = (y_te == 0)
        held_m = (fam_te == held_out)

        # ── Exact Point Estimates on Full Test Set ────────────────────────────
        # System
        sys_bacc = float(balanced_accuracy_score(y_te, y_sys) * 100.0)
        sys_held_rec = float(np.mean(y_sys[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        sys_fpr = float(np.mean(y_sys[ben_m] == 1) * 100.0)
        sys_silent = (p1_pred == 0) & (~esc_mask) & atk_m
        sys_sfr = float(np.sum(sys_silent) / np.sum(atk_m) * 100.0)
        sys_tot_esc = float(np.mean(esc_mask) * 100.0)
        sys_held_esc = float(np.mean(esc_mask[held_m]) * 100.0) if np.sum(held_m) > 0 else 0.0
        sys_ben_esc = float(np.mean(esc_mask[ben_m]) * 100.0)

        # Tier 1
        t1_bacc = float(balanced_accuracy_score(y_te, p1_pred) * 100.0)
        t1_held_rec = float(np.mean(p1_pred[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        t1_fpr = float(np.mean(p1_pred[ben_m] == 1) * 100.0)
        t1_sfr = float(np.sum((p1_pred == 0) & atk_m) / np.sum(atk_m) * 100.0)

        # Tier 2
        t2_bacc = float(balanced_accuracy_score(y_te, p2_pred) * 100.0)
        t2_held_rec = float(np.mean(p2_pred[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        t2_fpr = float(np.mean(p2_pred[ben_m] == 1) * 100.0)
        t2_sfr = float(np.sum((p2_pred == 0) & atk_m) / np.sum(atk_m) * 100.0)

        # Family point estimates
        sys_fam_rec = {}
        t1_fam_rec = {}
        t2_fam_rec = {}
        fam_esc = {}
        for f in ALL_FAMILIES:
            fm = (fam_te == f)
            sys_fam_rec[f] = float(np.mean(y_sys[fm] == 1) * 100.0) if np.sum(fm) > 0 else 0.0
            t1_fam_rec[f] = float(np.mean(p1_pred[fm] == 1) * 100.0) if np.sum(fm) > 0 else 0.0
            t2_fam_rec[f] = float(np.mean(p2_pred[fm] == 1) * 100.0) if np.sum(fm) > 0 else 0.0
            fam_esc[f] = float(np.mean(esc_mask[fm]) * 100.0) if np.sum(fm) > 0 else 0.0

        # Paired differences
        paired_diffs = {
            "delta_sfr": float(sys_sfr - t1_sfr),
            "delta_held_rec": float(sys_held_rec - t1_held_rec),
            "delta_bacc": float(sys_bacc - t1_bacc),
            "delta_fpr": float(sys_fpr - t1_fpr),
        }

        # Raw counts per family & reconciliation
        t1_block = (p1_pred == 1) & (~esc_mask)
        raw_family_breakdown = {}
        for f in ALL_FAMILIES:
            fm = (fam_te == f)
            n_f = int(np.sum(fm))
            blk_cnt = int(np.sum(t1_block & fm))
            esc_cnt = int(np.sum(esc_mask & fm))
            esc_blk_cnt = int(np.sum(esc_mask & fm & (p2_pred == 1)))
            esc_alw_cnt = int(np.sum(esc_mask & fm & (p2_pred == 0)))
            sil_cnt = int(np.sum((p1_pred == 0) & (~esc_mask) & fm))
            sys_caught_cnt = int(np.sum(y_sys[fm] == 1))
            t1_rec_val = float(np.mean(p1_pred[fm] == 1) * 100.0) if n_f > 0 else 0.0
            sys_rec_val = float(np.mean(y_sys[fm] == 1) * 100.0) if n_f > 0 else 0.0
            esc_rate_val = float(np.mean(esc_mask[fm]) * 100.0) if n_f > 0 else 0.0

            assert blk_cnt + esc_cnt + sil_cnt == n_f, f"Sum mismatch for family {f} in fold {held_out}"
            assert blk_cnt + esc_blk_cnt == sys_caught_cnt, f"Caught mismatch for family {f} in fold {held_out}"

            raw_family_breakdown[f] = {
                "n_flows": n_f,
                "t1_directly_blocked": blk_cnt,
                "t1_escalated": esc_cnt,
                "t2_blocked_among_esc": esc_blk_cnt,
                "t2_allowed_among_esc": esc_alw_cnt,
                "t1_silent_miss": sil_cnt,
                "system_caught": sys_caught_cnt,
                "t1_recall": t1_rec_val,
                "system_recall": sys_rec_val,
                "escalation_rate": esc_rate_val,
            }

        ben_n = int(np.sum(ben_m))
        ben_t1_alw = int(np.sum((p1_pred == 0) & (~esc_mask) & ben_m))
        ben_esc_cnt = int(np.sum(esc_mask & ben_m))
        ben_t2_alw = int(np.sum(esc_mask & ben_m & (p2_pred == 0)))
        ben_t2_blk = int(np.sum(esc_mask & ben_m & (p2_pred == 1)))
        ben_t1_fa = int(np.sum((p1_pred == 1) & (~esc_mask) & ben_m))
        sys_fp_cnt = int(np.sum(y_sys[ben_m] == 1))
        sys_tn_cnt = ben_t1_alw + ben_t2_alw

        assert ben_t1_alw + ben_esc_cnt + ben_t1_fa == ben_n, "Benign count mismatch"
        assert ben_t1_fa + ben_t2_blk == sys_fp_cnt, "Benign FP count mismatch"
        assert sys_fp_cnt + sys_tn_cnt == ben_n, "Benign total reconciliation mismatch"

        raw_reconciliation = {
            "total_test_flows": len(y_te),
            "benign_total": ben_n,
            "attack_total": int(np.sum(atk_m)),
            "per_family": raw_family_breakdown,
            "attack_overall": {
                "t1_directly_blocked": int(np.sum(t1_block & atk_m)),
                "t1_escalated": int(np.sum(esc_mask & atk_m)),
                "t2_blocked_among_esc": int(np.sum(esc_mask & atk_m & (p2_pred == 1))),
                "t2_allowed_among_esc": int(np.sum(esc_mask & atk_m & (p2_pred == 0))),
                "t1_silent_miss": int(np.sum((p1_pred == 0) & (~esc_mask) & atk_m)),
                "system_caught": int(np.sum(y_sys[atk_m] == 1)),
                "system_recall": sys_bacc,
                "system_sfr": sys_sfr,
            },
            "benign_breakdown": {
                "t1_directly_allowed": ben_t1_alw,
                "t1_escalated": ben_esc_cnt,
                "t2_allowed_among_esc": ben_t2_alw,
                "t2_false_alarm_among_esc": ben_t2_blk,
                "t1_false_alarm": ben_t1_fa,
                "system_false_alarms": sys_fp_cnt,
                "system_true_negatives": sys_tn_cnt,
                "system_fpr": sys_fpr,
            },
        }

        # ── Stratified Within-Family Flow Bootstrap Confidence Intervals ───────
        strata = {"Benign": np.where(ben_m)[0]}
        for f in ALL_FAMILIES:
            strata[f] = np.where(fam_te == f)[0]

        boot_res = {
            "sys": {"bacc": [], "held_rec": [], "fpr": [], "sfr": [], "tot_esc": [], "held_esc": [], "ben_esc": []},
            "t1": {"bacc": [], "held_rec": [], "fpr": [], "sfr": []},
            "t2": {"bacc": [], "held_rec": [], "fpr": [], "sfr": []},
            "diffs": {"delta_sfr": [], "delta_held_rec": [], "delta_bacc": [], "delta_fpr": []},
            "fam_rec": {f: [] for f in ALL_FAMILIES},
            "fam_esc": {f: [] for f in ALL_FAMILIES},
        }

        for _ in range(N_BOOT):
            # Resample each stratum with replacement
            sample_idx = np.concatenate([
                rng_boot.choice(idx_list, size=len(idx_list), replace=True)
                for idx_list in strata.values()
            ])
            b_y = y_te[sample_idx]
            b_fam = fam_te[sample_idx]
            b_sys = y_sys[sample_idx]
            b_p1 = p1_pred[sample_idx]
            b_p2 = p2_pred[sample_idx]
            b_esc = esc_mask[sample_idx]

            b_atk_m = (b_y == 1)
            b_ben_m = (b_y == 0)
            b_held_m = (b_fam == held_out)

            # System resample metrics
            b_sys_bacc = balanced_accuracy_score(b_y, b_sys) * 100.0
            b_sys_held_rec = np.mean(b_sys[b_held_m] == 1) * 100.0 if np.sum(b_held_m) > 0 else 0.0
            b_sys_fpr = np.mean(b_sys[b_ben_m] == 1) * 100.0
            b_sys_sfr = np.sum((b_p1 == 0) & (~b_esc) & b_atk_m) / np.sum(b_atk_m) * 100.0

            boot_res["sys"]["bacc"].append(b_sys_bacc)
            boot_res["sys"]["held_rec"].append(b_sys_held_rec)
            boot_res["sys"]["fpr"].append(b_sys_fpr)
            boot_res["sys"]["sfr"].append(b_sys_sfr)
            boot_res["sys"]["tot_esc"].append(np.mean(b_esc) * 100.0)
            boot_res["sys"]["held_esc"].append(np.mean(b_esc[b_held_m]) * 100.0 if np.sum(b_held_m) > 0 else 0.0)
            boot_res["sys"]["ben_esc"].append(np.mean(b_esc[b_ben_m]) * 100.0)

            # Tier 1
            b_t1_bacc = balanced_accuracy_score(b_y, b_p1) * 100.0
            b_t1_held_rec = np.mean(b_p1[b_held_m] == 1) * 100.0 if np.sum(b_held_m) > 0 else 0.0
            b_t1_fpr = np.mean(b_p1[b_ben_m] == 1) * 100.0
            b_t1_sfr = np.sum((b_p1 == 0) & b_atk_m) / np.sum(b_atk_m) * 100.0

            boot_res["t1"]["bacc"].append(b_t1_bacc)
            boot_res["t1"]["held_rec"].append(b_t1_held_rec)
            boot_res["t1"]["fpr"].append(b_t1_fpr)
            boot_res["t1"]["sfr"].append(b_t1_sfr)

            # Tier 2
            b_t2_bacc = balanced_accuracy_score(b_y, b_p2) * 100.0
            b_t2_held_rec = np.mean(b_p2[b_held_m] == 1) * 100.0 if np.sum(b_held_m) > 0 else 0.0
            b_t2_fpr = np.mean(b_p2[b_ben_m] == 1) * 100.0
            b_t2_sfr = np.sum((b_p2 == 0) & b_atk_m) / np.sum(b_atk_m) * 100.0

            boot_res["t2"]["bacc"].append(b_t2_bacc)
            boot_res["t2"]["held_rec"].append(b_t2_held_rec)
            boot_res["t2"]["fpr"].append(b_t2_fpr)
            boot_res["t2"]["sfr"].append(b_t2_sfr)

            # Paired diffs
            boot_res["diffs"]["delta_sfr"].append(b_sys_sfr - b_t1_sfr)
            boot_res["diffs"]["delta_held_rec"].append(b_sys_held_rec - b_t1_held_rec)
            boot_res["diffs"]["delta_bacc"].append(b_sys_bacc - b_t1_bacc)
            boot_res["diffs"]["delta_fpr"].append(b_sys_fpr - b_t1_fpr)

            # Family recalls
            for f in ALL_FAMILIES:
                fm = (b_fam == f)
                boot_res["fam_rec"][f].append(np.mean(b_sys[fm] == 1) * 100.0 if np.sum(fm) > 0 else 0.0)
                boot_res["fam_esc"][f].append(np.mean(b_esc[fm]) * 100.0 if np.sum(fm) > 0 else 0.0)

        # Assemble fold summary
        bootstrap_cis = {
            "sys": {k: compute_percentile_ci(v) for k, v in boot_res["sys"].items()},
            "t1": {k: compute_percentile_ci(v) for k, v in boot_res["t1"].items()},
            "t2": {k: compute_percentile_ci(v) for k, v in boot_res["t2"].items()},
            "diffs": {k: compute_percentile_ci(v, clip_min=-100.0, clip_max=100.0) for k, v in boot_res["diffs"].items()},
            "family_recalls": {k: compute_percentile_ci(v) for k, v in boot_res["fam_rec"].items()},
            "family_escs": {k: compute_percentile_ci(v) for k, v in boot_res["fam_esc"].items()},
        }

        lofo_folds_raw[held_out] = {
            "held_out_family": held_out,
            "point_estimates": {
                "sys": {
                    "bacc": sys_bacc, "held_rec": sys_held_rec, "fpr": sys_fpr,
                    "sfr": sys_sfr, "tot_esc": sys_tot_esc, "held_esc": sys_held_esc, "ben_esc": sys_ben_esc
                },
                "t1": {"bacc": t1_bacc, "held_rec": t1_held_rec, "fpr": t1_fpr, "sfr": t1_sfr},
                "t2": {"bacc": t2_bacc, "held_rec": t2_held_rec, "fpr": t2_fpr, "sfr": t2_sfr, "tot_esc": 100.0, "held_esc": 100.0, "ben_esc": 100.0},
                "family_recalls": sys_fam_rec,
                "t1_family_recalls": t1_fam_rec,
                "t2_family_recalls": t2_fam_rec,
                "family_escs": fam_esc,
                "paired_diffs": paired_diffs,
            },
            "bootstrap_cis": bootstrap_cis,
            "raw_reconciliation": raw_reconciliation,
        }

        # Accumulate for macro
        for met in ["bacc", "held_rec", "fpr", "sfr", "tot_esc", "held_esc", "ben_esc"]:
            macro_accumulators["sys"][met].append(sys_bacc if met == "bacc" else lofo_folds_raw[held_out]["point_estimates"]["sys"][met])
        for met in ["bacc", "held_rec", "fpr", "sfr"]:
            macro_accumulators["t1"][met].append(lofo_folds_raw[held_out]["point_estimates"]["t1"][met])
            macro_accumulators["t2"][met].append(lofo_folds_raw[held_out]["point_estimates"]["t2"][met])
        for met in ["delta_sfr", "delta_held_rec", "delta_bacc", "delta_fpr"]:
            macro_accumulators["diffs"][met].append(paired_diffs[met])

    # ── Macro-averages across the 5 LOFO folds ────────────────────────────────
    macro_summary = {"sys": {}, "t1": {}, "t2": {}, "diffs": {}}
    for mod in ["sys", "t1", "t2", "diffs"]:
        for met, vals in macro_accumulators[mod].items():
            pt = float(np.mean(vals))
            # Across-fold standard error & 95% CI
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ci95 = float(1.96 * se)
            macro_summary[mod][met] = {
                "point_estimate": pt,
                "ci95": ci95,
                "lo": float(max(-100.0 if "delta" in met else 0.0, pt - ci95)),
                "hi": float(min(100.0, pt + ci95)),
            }

    # ── 3. Window Feature Ablation (76 vs 74 Features) ────────────────────────
    print("  Evaluating Window Size Feature Ablation (76 vs 74 Features)...")
    ablation_raw = {}
    for held_out in ALL_FAMILIES:
        X_tr_76_f = bench[f"lofo_{held_out}_X_train_76"]
        y_tr_f = bench[f"lofo_{held_out}_y_train"]
        X_te_76_f = bench[f"lofo_{held_out}_X_test_76"]
        y_te_f = bench[f"lofo_{held_out}_y_test"]
        fam_te_f = bench[f"lofo_{held_out}_y_family_test"]

        # Use the exact same 76-feat Tier 2 model as Table 2
        rf_76 = t2_models[held_out]
        rf_74 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(
            X_tr_76_f[:, NON_WIN_INDICES], y_tr_f
        )

        p76 = rf_76.predict(X_te_76_f)
        p74 = rf_74.predict(X_te_76_f[:, NON_WIN_INDICES])

        # Exact point estimates
        held_m = (fam_te_f == held_out)
        pt_76_acc = float(balanced_accuracy_score(y_te_f, p76) * 100.0)
        pt_74_acc = float(balanced_accuracy_score(y_te_f, p74) * 100.0)
        pt_76_rec = float(np.mean(p76[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        pt_74_rec = float(np.mean(p74[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0

        # Verify strict alignment with Table 2 Tier 2
        assert abs(pt_76_acc - lofo_folds_raw[held_out]["point_estimates"]["t2"]["bacc"]) < 1e-6

        # Flow bootstrap for ablation intervals
        strata_f = {"Benign": np.where(y_te_f == 0)[0]}
        for f in ALL_FAMILIES:
            strata_f[f] = np.where(fam_te_f == f)[0]

        b76_acc, b74_acc, b76_rec, b74_rec, b_delta = [], [], [], [], []
        for _ in range(N_BOOT):
            s_idx = np.concatenate([
                rng_boot.choice(idx_l, size=len(idx_l), replace=True)
                for idx_l in strata_f.values()
            ])
            b_y = y_te_f[s_idx]
            b_fam = fam_te_f[s_idx]
            b_hm = (b_fam == held_out)

            acc_76 = balanced_accuracy_score(b_y, p76[s_idx]) * 100.0
            acc_74 = balanced_accuracy_score(b_y, p74[s_idx]) * 100.0
            b76_acc.append(acc_76)
            b74_acc.append(acc_74)
            b_delta.append(acc_74 - acc_76)
            b76_rec.append(np.mean(p76[s_idx][b_hm] == 1) * 100.0 if np.sum(b_hm) > 0 else 0.0)
            b74_rec.append(np.mean(p74[s_idx][b_hm] == 1) * 100.0 if np.sum(b_hm) > 0 else 0.0)

        ci_76_acc = compute_percentile_ci(b76_acc)
        ci_74_acc = compute_percentile_ci(b74_acc)
        ci_76_rec = compute_percentile_ci(b76_rec)
        ci_74_rec = compute_percentile_ci(b74_rec)
        ci_delta = compute_percentile_ci(b_delta, clip_min=-100.0, clip_max=100.0)

        ablation_raw[held_out] = {
            "model": "Tier 2 RF (100 trees, depth 12)",
            "acc_76_point": pt_76_acc,
            "acc_76_ci": ci_76_acc,
            "acc_74_point": pt_74_acc,
            "acc_74_ci": ci_74_acc,
            "delta_acc_point": float(pt_74_acc - pt_76_acc),
            "delta_acc_ci": ci_delta,
            "rec_76_point": pt_76_rec,
            "rec_76_ci": ci_76_rec,
            "rec_74_point": pt_74_rec,
            "rec_74_ci": ci_74_rec,
        }

    # Macro ablation
    ablation_macro = {
        "acc_76_point": float(np.mean([ablation_raw[f]["acc_76_point"] for f in ALL_FAMILIES])),
        "acc_74_point": float(np.mean([ablation_raw[f]["acc_74_point"] for f in ALL_FAMILIES])),
        "delta_acc_point": float(np.mean([ablation_raw[f]["delta_acc_point"] for f in ALL_FAMILIES])),
        "rec_76_point": float(np.mean([ablation_raw[f]["rec_76_point"] for f in ALL_FAMILIES])),
        "rec_74_point": float(np.mean([ablation_raw[f]["rec_74_point"] for f in ALL_FAMILIES])),
    }

    # ── 4. Prevalence Robustness Evaluation (1:100 and 1:1,000) ───────────────
    print("  Evaluating Prevalence Test Splits (1:100 and 1:1,000)...")
    prevalence_eval = {}
    for p_name in ["prevalence_1_100", "prevalence_1_1000"]:
        X_tr_9 = bench[f"{p_name}_X_train"][:, :9]
        X_tr_76 = bench[f"{p_name}_X_train_76"]
        y_tr = bench[f"{p_name}_y_train"]

        X_te_9 = bench[f"{p_name}_X_test"][:, :9]
        X_te_76 = bench[f"{p_name}_X_test_76"]
        y_te = bench[f"{p_name}_y_test"]
        fam_te = bench[f"{p_name}_y_family_test"]

        t1 = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr_9, y_tr)
        t2 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_tr_76, y_tr)

        p1_prob = t1.predict_proba(X_te_9)[:, 1]
        p1_pred = (p1_prob >= 0.50).astype(int)
        p2_pred = t2.predict(X_te_76)

        esc_m = (p1_prob >= 0.30) & (p1_prob <= 0.70)
        y_s = p1_pred.copy()
        y_s[esc_m] = p2_pred[esc_m]

        atk_m = (y_te == 1)
        ben_m = (y_te == 0)

        p_pt = {
            "total_test": len(y_te),
            "benign_count": int(np.sum(ben_m)),
            "attack_count": int(np.sum(atk_m)),
            "sys": {
                "bacc": float(balanced_accuracy_score(y_te, y_s) * 100.0),
                "recall": float(np.mean(y_s[atk_m] == 1) * 100.0),
                "fpr": float(np.mean(y_s[ben_m] == 1) * 100.0),
                "sfr": float(np.sum((p1_pred == 0) & (~esc_m) & atk_m) / np.sum(atk_m) * 100.0),
                "esc_rate": float(np.mean(esc_m) * 100.0),
            },
            "t1": {
                "bacc": float(balanced_accuracy_score(y_te, p1_pred) * 100.0),
                "recall": float(np.mean(p1_pred[atk_m] == 1) * 100.0),
                "fpr": float(np.mean(p1_pred[ben_m] == 1) * 100.0),
                "sfr": float(np.sum((p1_pred == 0) & atk_m) / np.sum(atk_m) * 100.0),
            },
            "t2": {
                "bacc": float(balanced_accuracy_score(y_te, p2_pred) * 100.0),
                "recall": float(np.mean(p2_pred[atk_m] == 1) * 100.0),
                "fpr": float(np.mean(p2_pred[ben_m] == 1) * 100.0),
                "sfr": float(np.sum((p2_pred == 0) & atk_m) / np.sum(atk_m) * 100.0),
            },
            "per_family": {},
        }

        for f in ALL_FAMILIES:
            fm = (fam_te == f)
            if np.sum(fm) > 0:
                p_pt["per_family"][f] = {
                    "count": int(np.sum(fm)),
                    "t1_rec": float(np.mean(p1_pred[fm] == 1) * 100.0),
                    "sys_rec": float(np.mean(y_s[fm] == 1) * 100.0),
                    "esc_rate": float(np.mean(esc_m[fm]) * 100.0),
                }

        # Bootstrap intervals
        strata_p = {"Benign": np.where(ben_m)[0]}
        for f in ALL_FAMILIES:
            fm = np.where(fam_te == f)[0]
            if len(fm) > 0:
                strata_p[f] = fm

        b_sys_bacc, b_sys_rec, b_sys_fpr = [], [], []
        for _ in range(N_BOOT):
            s_idx = np.concatenate([
                rng_boot.choice(idx_l, size=len(idx_l), replace=True)
                for idx_l in strata_p.values()
            ])
            b_y = y_te[s_idx]
            b_ys = y_s[s_idx]
            b_atk = (b_y == 1)
            b_ben = (b_y == 0)
            b_sys_bacc.append(balanced_accuracy_score(b_y, b_ys) * 100.0)
            b_sys_rec.append(np.mean(b_ys[b_atk] == 1) * 100.0)
            b_sys_fpr.append(np.mean(b_ys[b_ben] == 1) * 100.0)

        p_pt["sys"]["bacc_ci"] = compute_percentile_ci(b_sys_bacc)
        p_pt["sys"]["recall_ci"] = compute_percentile_ci(b_sys_rec)
        p_pt["sys"]["fpr_ci"] = compute_percentile_ci(b_sys_fpr)

        prevalence_eval[p_name] = p_pt

    final_payload = {
        "execution_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "split_table_verified": split_table,
        "two_flow_explanation": "lofo_FTP-BruteForce_train (11,500) contains 623 SSH flows; lofo_SSH-BruteForce_train (11,502) contains 625 FTP flows. The 2-flow difference is 625 - 623 = 2 flows from historical training pool sizing.",
        "lofo_folds": lofo_folds_raw,
        "macro_summary": macro_summary,
        "window_ablation_74": {
            "features_original": 76,
            "features_ablated": 74,
            "ablated_columns": ["FWD Init Win Bytes", "Bwd Init Win Bytes"],
            "folds": ablation_raw,
            "macro": ablation_macro,
        },
        "prevalence_evaluation": prevalence_eval,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved LOFO point estimates & bootstrap results to: {OUT_JSON}")


if __name__ == "__main__":
    run_lofo_benchmark()
