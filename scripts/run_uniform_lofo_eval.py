#!/usr/bin/env python3
"""
scripts/run_uniform_lofo_eval.py
────────────────────────────────
Evaluates the Leave-One-Family-Out (LOFO) benchmark on data/processed/benchmark_uniform_dataset.npz:
  1. Main LOFO Table on Uniform Benchmark:
     - 5 folds: FTP-BruteForce, SSH-BruteForce, DoS-GoldenEye, DoS-Hulk, DoS-Slowloris
     - Models: Integrated Two-Tier System, Tier 1 Standalone, Tier 2 Standalone
     - Exact point estimates evaluated on the un-resampled full test set.
     - Confidence intervals computed via stratified within-family flow bootstrap (1,000 resamples).
     - Full flow reconciliation (attack flows per family, blocked, escalated, silent).
  2. Paired Differences:
     - Evaluated with MACRO t-interval OVER FOLDS ONLY (df=4, t_crit=2.7764).
  3. Window Feature Ablation (76 vs 74 Features):
     - Point estimates strictly aligned with Table 2 Tier 2 Standalone.
  4. Prevalence Robustness Evaluation (1:100 and 1:1,000):
     - Resampling attacks AND benign with replacement across 1,000 draws.
     - Wilson score intervals for FPR.
     - Empirical percentile CIs for Precision and Recall.
  5. Serializes authoritative results to results/lofo/lofo_uniform_raw.json
     and updates results/lofo/lofo_block_bootstrap_raw.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_guard import ALL_ALLOWLISTED_CIC_FEATURES

UNIFORM_NPZ = "data/processed/benchmark_uniform_dataset.npz"
OUT_JSON = "results/lofo/lofo_uniform_raw.json"
COMPAT_JSON = "results/lofo/lofo_block_bootstrap_raw.json"
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


def compute_macro_t_interval_over_folds(vals: List[float], alpha: float = 0.05) -> Dict[str, float]:
    """
    Compute Student's t 95% confidence interval strictly over the K=5 fold values (df = K-1 = 4).
    t_crit = 2.776445.
    """
    arr = np.array(vals, dtype=float)
    k = len(arr)
    assert k == 5, f"Expected exactly 5 folds for macro t-interval, got {k}"
    mean_val = float(np.mean(arr))
    s_folds = float(np.std(arr, ddof=1))
    se_macro = float(s_folds / np.sqrt(k))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 1))
    hw = float(t_crit * se_macro)
    return {
        "mean": mean_val,
        "sample_std": s_folds,
        "se": se_macro,
        "df": k - 1,
        "t_crit": t_crit,
        "half_width": hw,
        "lo": mean_val - hw,
        "hi": mean_val + hw,
    }


def compute_wilson_score_interval(x: int, n: int, z: float = 1.959964) -> Dict[str, float]:
    """
    Computes the Wilson score interval for a binomial proportion x/n.
    Returns point estimate, lower, and upper bounds (in percent).
    """
    if n == 0:
        return {"pt": 0.0, "lo": 0.0, "hi": 0.0, "half_width": 0.0}
    p_hat = x / n
    denominator = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denominator
    factor = z * np.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4 * n)) / n) / denominator
    lo = max(0.0, float((center - factor) * 100.0))
    hi = min(100.0, float((center + factor) * 100.0))
    pt = float(p_hat * 100.0)
    return {
        "pt": pt,
        "lo": lo,
        "hi": hi,
        "half_width": float((hi - lo) / 2.0),
    }


def run_uniform_lofo():
    print("=" * 80)
    print(" RUNNING UNIFORM BENCHMARK LOFO EVALUATION WITH MACRO t-INTERVALS OVER FOLDS")
    print("=" * 80)

    if not os.path.exists(UNIFORM_NPZ):
        raise FileNotFoundError(f"Missing uniform benchmark dataset: {UNIFORM_NPZ}")

    bench = np.load(UNIFORM_NPZ)
    rng_boot = np.random.RandomState(42)

    # ── 1. Component Sums Verification ────────────────────────────────────────
    split_names = [
        "ftp_only_train", "ftp_only_test",
        "lofo_FTP-BruteForce_train", "lofo_SSH-BruteForce_train",
        "lofo_DoS-GoldenEye_train", "lofo_DoS-Hulk_train", "lofo_DoS-Slowloris_train",
    ]
    verified_splits = {}
    for sname in split_names:
        y_tr = bench[f"{sname[:sname.rfind('_')]}_y_{sname[sname.rfind('_')+1:]}"]
        y_fam = bench.get(f"{sname[:sname.rfind('_')]}_y_family_{sname[sname.rfind('_')+1:]}")
        total = len(y_tr)
        ben_cnt = int(np.sum(y_tr == 0))
        atk_cnt = int(np.sum(y_tr == 1))
        f_counts = {}
        if y_fam is not None:
            for f in ALL_FAMILIES:
                f_counts[f] = int(np.sum(y_fam == f))
        verified_splits[sname] = {
            "total_flows": total,
            "benign_flows": ben_cnt,
            "attack_flows": atk_cnt,
            "family_counts": f_counts,
        }

    # ── 2. Run Main LOFO Table ────────────────────────────────────────────────
    lofo_folds_raw = {}
    t2_models = {}
    macro_accumulators = {
        "sys": {m: [] for m in ["bacc", "held_rec", "fpr", "sfr", "tot_esc", "held_esc", "ben_esc"]},
        "t1": {m: [] for m in ["bacc", "held_rec", "fpr", "sfr"]},
        "t2": {m: [] for m in ["bacc", "held_rec", "fpr", "sfr"]},
        "diffs": {m: [] for m in ["delta_sfr", "delta_held_rec", "delta_bacc", "delta_fpr"]},
    }

    for held_out in ALL_FAMILIES:
        print(f"\nProcessing Uniform LOFO Fold: Held-out {held_out}...")
        X_tr_9 = bench[f"lofo_{held_out}_X_train"][:, :9]
        X_tr_76 = bench[f"lofo_{held_out}_X_train_76"]
        y_tr = bench[f"lofo_{held_out}_y_train"]

        X_te_9 = bench[f"lofo_{held_out}_X_test"][:, :9]
        X_te_76 = bench[f"lofo_{held_out}_X_test_76"]
        y_te = bench[f"lofo_{held_out}_y_test"]
        fam_te = bench[f"lofo_{held_out}_y_family_test"]

        # Models
        t1 = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr_9, y_tr)
        t2 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_tr_76, y_tr)
        t2_models[held_out] = t2

        p1_prob = t1.predict_proba(X_te_9)[:, 1]
        p1_pred = (p1_prob >= 0.50).astype(int)
        p2_pred = t2.predict(X_te_76)

        # Standard RF Margin escalation: 0.30 <= prob <= 0.70
        esc_mask = (p1_prob >= 0.30) & (p1_prob <= 0.70)
        y_sys = p1_pred.copy()
        y_sys[esc_mask] = p2_pred[esc_mask]

        atk_m = (y_te == 1)
        ben_m = (y_te == 0)
        held_m = (fam_te == held_out)

        # Exact Point Estimates on Full Test Set
        sys_bacc = float(balanced_accuracy_score(y_te, y_sys) * 100.0)
        sys_held_rec = float(np.mean(y_sys[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        sys_fpr = float(np.mean(y_sys[ben_m] == 1) * 100.0)
        sys_silent = (p1_pred == 0) & (~esc_mask) & atk_m
        sys_sfr = float(np.sum(sys_silent) / np.sum(atk_m) * 100.0)
        sys_tot_esc = float(np.mean(esc_mask) * 100.0)
        sys_held_esc = float(np.mean(esc_mask[held_m]) * 100.0) if np.sum(held_m) > 0 else 0.0
        sys_ben_esc = float(np.mean(esc_mask[ben_m]) * 100.0)

        t1_bacc = float(balanced_accuracy_score(y_te, p1_pred) * 100.0)
        t1_held_rec = float(np.mean(p1_pred[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        t1_fpr = float(np.mean(p1_pred[ben_m] == 1) * 100.0)
        t1_sfr = float(np.sum((p1_pred == 0) & atk_m) / np.sum(atk_m) * 100.0)

        t2_bacc = float(balanced_accuracy_score(y_te, p2_pred) * 100.0)
        t2_held_rec = float(np.mean(p2_pred[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        t2_fpr = float(np.mean(p2_pred[ben_m] == 1) * 100.0)
        t2_sfr = float(np.sum((p2_pred == 0) & atk_m) / np.sum(atk_m) * 100.0)

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

        paired_diffs = {
            "delta_sfr": float(sys_sfr - t1_sfr),
            "delta_held_rec": float(sys_held_rec - t1_held_rec),
            "delta_bacc": float(sys_bacc - t1_bacc),
            "delta_fpr": float(sys_fpr - t1_fpr),
        }

        # Flow reconciliation
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

            assert blk_cnt + esc_cnt + sil_cnt == n_f
            assert blk_cnt + esc_blk_cnt == sys_caught_cnt

            raw_family_breakdown[f] = {
                "n_flows": n_f,
                "t1_directly_blocked": blk_cnt,
                "t1_escalated": esc_cnt,
                "t2_blocked_among_esc": esc_blk_cnt,
                "t2_allowed_among_esc": esc_alw_cnt,
                "t1_silent_miss": sil_cnt,
                "system_caught": sys_caught_cnt,
                "t1_recall": t1_fam_rec[f],
                "system_recall": sys_fam_rec[f],
                "escalation_rate": fam_esc[f],
            }

        ben_n = int(np.sum(ben_m))
        ben_t1_alw = int(np.sum((p1_pred == 0) & (~esc_mask) & ben_m))
        ben_esc_cnt = int(np.sum(esc_mask & ben_m))
        ben_t2_alw = int(np.sum(esc_mask & ben_m & (p2_pred == 0)))
        ben_t2_blk = int(np.sum(esc_mask & ben_m & (p2_pred == 1)))
        ben_t1_fa = int(np.sum((p1_pred == 1) & (~esc_mask) & ben_m))
        sys_fp_cnt = int(np.sum(y_sys[ben_m] == 1))
        sys_tn_cnt = ben_t1_alw + ben_t2_alw

        assert ben_t1_alw + ben_esc_cnt + ben_t1_fa == ben_n
        assert ben_t1_fa + ben_t2_blk == sys_fp_cnt
        assert sys_fp_cnt + sys_tn_cnt == ben_n

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
                "system_recall": float(np.mean(y_sys[atk_m] == 1) * 100.0),
            },
            "benign_reconciliation": {
                "t1_directly_allowed": ben_t1_alw,
                "t1_escalated": ben_esc_cnt,
                "t2_allowed_among_esc": ben_t2_alw,
                "t2_blocked_among_esc (FP)": ben_t2_blk,
                "t1_direct_false_alarms": ben_t1_fa,
                "system_false_alarms": sys_fp_cnt,
                "system_true_negatives": sys_tn_cnt,
            },
        }

        # Stratified within-family bootstrap for confidence intervals (1,000 resamples)
        strata = {"Benign": np.where(ben_m)[0]}
        for f in ALL_FAMILIES:
            strata[f] = np.where(fam_te == f)[0]

        boot_res = {
            "sys": {k: [] for k in ["bacc", "held_rec", "fpr", "sfr", "tot_esc", "held_esc", "ben_esc"]},
            "t1": {k: [] for k in ["bacc", "held_rec", "fpr", "sfr"]},
            "t2": {k: [] for k in ["bacc", "held_rec", "fpr", "sfr"]},
            "diffs": {k: [] for k in ["delta_sfr", "delta_held_rec", "delta_bacc", "delta_fpr"]},
            "fam_rec": {f: [] for f in ALL_FAMILIES},
            "fam_esc": {f: [] for f in ALL_FAMILIES},
        }

        for _ in range(N_BOOT):
            s_idx = np.concatenate([
                rng_boot.choice(idx_l, size=len(idx_l), replace=True)
                for idx_l in strata.values()
            ])
            b_y = y_te[s_idx]
            b_fam = fam_te[s_idx]
            b_p1 = p1_pred[s_idx]
            b_p2 = p2_pred[s_idx]
            b_esc = esc_mask[s_idx]
            b_sys = y_sys[s_idx]

            b_atk_m = (b_y == 1)
            b_ben_m = (b_y == 0)
            b_held_m = (b_fam == held_out)

            # System
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

            boot_res["diffs"]["delta_sfr"].append(b_sys_sfr - b_t1_sfr)
            boot_res["diffs"]["delta_held_rec"].append(b_sys_held_rec - b_t1_held_rec)
            boot_res["diffs"]["delta_bacc"].append(b_sys_bacc - b_t1_bacc)
            boot_res["diffs"]["delta_fpr"].append(b_sys_fpr - b_t1_fpr)

            for f in ALL_FAMILIES:
                fm = (b_fam == f)
                boot_res["fam_rec"][f].append(np.mean(b_sys[fm] == 1) * 100.0 if np.sum(fm) > 0 else 0.0)
                boot_res["fam_esc"][f].append(np.mean(b_esc[fm]) * 100.0 if np.sum(fm) > 0 else 0.0)

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
            macro_accumulators["sys"][met].append(lofo_folds_raw[held_out]["point_estimates"]["sys"][met])
        for met in ["bacc", "held_rec", "fpr", "sfr"]:
            macro_accumulators["t1"][met].append(lofo_folds_raw[held_out]["point_estimates"]["t1"][met])
            macro_accumulators["t2"][met].append(lofo_folds_raw[held_out]["point_estimates"]["t2"][met])
        for met in ["delta_sfr", "delta_held_rec", "delta_bacc", "delta_fpr"]:
            macro_accumulators["diffs"][met].append(paired_diffs[met])

    # ── Macro-averages across the 5 LOFO folds using Macro t-interval over folds ──
    macro_summary = {"sys": {}, "t1": {}, "t2": {}, "diffs": {}}
    for mod in ["sys", "t1", "t2"]:
        for met, vals in macro_accumulators[mod].items():
            t_res = compute_macro_t_interval_over_folds(vals)
            macro_summary[mod][met] = {
                "point_estimate": t_res["mean"],
                "se": t_res["se"],
                "df": t_res["df"],
                "t_crit": t_res["t_crit"],
                "ci95": t_res["half_width"],
                "lo": max(0.0, t_res["lo"]),
                "hi": min(100.0, t_res["hi"]),
            }

    # Paired differences: strictly macro t-interval over the 5 folds only
    for met, vals in macro_accumulators["diffs"].items():
        t_res = compute_macro_t_interval_over_folds(vals)
        macro_summary["diffs"][met] = {
            "point_estimate": t_res["mean"],
            "sample_std": t_res["sample_std"],
            "se": t_res["se"],
            "df": t_res["df"],
            "t_crit": t_res["t_crit"],
            "ci95": t_res["half_width"],
            "lo": max(-100.0, t_res["lo"]),
            "hi": min(100.0, t_res["hi"]),
            "per_fold_values": vals,
        }

    # ── 3. Window Feature Ablation on Uniform Benchmark (76 vs 74 Features) ───
    print("\nEvaluating Window Feature Ablation (76 vs 74 Features) on Uniform Benchmark...")
    ablation_raw = {}
    for held_out in ALL_FAMILIES:
        X_tr_76_f = bench[f"lofo_{held_out}_X_train_76"]
        y_tr_f = bench[f"lofo_{held_out}_y_train"]
        X_te_76_f = bench[f"lofo_{held_out}_X_test_76"]
        y_te_f = bench[f"lofo_{held_out}_y_test"]
        fam_te_f = bench[f"lofo_{held_out}_y_family_test"]

        rf_76 = t2_models[held_out]
        rf_74 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(
            X_tr_76_f[:, NON_WIN_INDICES], y_tr_f
        )

        p76 = rf_76.predict(X_te_76_f)
        p74 = rf_74.predict(X_te_76_f[:, NON_WIN_INDICES])

        held_m = (fam_te_f == held_out)
        pt_76_acc = float(balanced_accuracy_score(y_te_f, p76) * 100.0)
        pt_74_acc = float(balanced_accuracy_score(y_te_f, p74) * 100.0)
        pt_76_rec = float(np.mean(p76[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        pt_74_rec = float(np.mean(p74[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0

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
            "acc_76_point": pt_76_acc,
            "acc_76_ci": ci_76_acc,
            "acc_74_point": pt_74_acc,
            "acc_74_ci": ci_74_acc,
            "delta_acc_point": pt_74_acc - pt_76_acc,
            "delta_acc_ci": ci_delta,
            "rec_76_point": pt_76_rec,
            "rec_76_ci": ci_76_rec,
            "rec_74_point": pt_74_rec,
            "rec_74_ci": ci_74_rec,
        }

    macro_ablation = {
        "acc_76_point": float(np.mean([ablation_raw[f]["acc_76_point"] for f in ALL_FAMILIES])),
        "acc_74_point": float(np.mean([ablation_raw[f]["acc_74_point"] for f in ALL_FAMILIES])),
        "delta_acc_point": float(np.mean([ablation_raw[f]["delta_acc_point"] for f in ALL_FAMILIES])),
        "rec_76_point": float(np.mean([ablation_raw[f]["rec_76_point"] for f in ALL_FAMILIES])),
        "rec_74_point": float(np.mean([ablation_raw[f]["rec_74_point"] for f in ALL_FAMILIES])),
    }

    # ── 4. Prevalence Evaluation (Resampling Attacks AND Benign) ───────────────
    print("\nEvaluating Prevalence with Resampling of Attacks AND Benign (1,000 Draws)...")
    X_te_9_full = bench["ftp_only_X_test"][:, :9]
    X_te_76_full = bench["ftp_only_X_test_76"]
    y_te_full = bench["ftp_only_y_test"]
    fam_te_full = bench["ftp_only_y_family_test"]

    ben_indices = np.where(y_te_full == 0)[0]
    atk_indices = np.where(y_te_full == 1)[0]
    n_ben_total = len(ben_indices)
    n_atk_total = len(atk_indices)

    # Base models trained on ftp_only_train
    X_tr_9_base = bench["ftp_only_X_train"][:, :9]
    X_tr_76_base = bench["ftp_only_X_train_76"]
    y_tr_base = bench["ftp_only_y_train"]

    t1_p = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr_9_base, y_tr_base)
    t2_p = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_tr_76_base, y_tr_base)

    prob1_all = t1_p.predict_proba(X_te_9_full)[:, 1]
    pred1_all = (prob1_all >= 0.50).astype(int)
    pred2_all = t2_p.predict(X_te_76_full)
    esc_all = (prob1_all >= 0.30) & (prob1_all <= 0.70)
    sys_all = pred1_all.copy()
    sys_all[esc_all] = pred2_all[esc_all]

    prevalence_subsampling = {}
    ratios = [
        {"name": "1:100", "n_attacks": max(5, int(n_ben_total * 0.01)), "ratio_val": 0.01},
        {"name": "1:1000", "n_attacks": max(5, int(n_ben_total * 0.001)), "ratio_val": 0.001},
    ]

    for r_info in ratios:
        r_name = r_info["name"]
        n_atk_sub = r_info["n_attacks"]
        print(f"  Running 1,000 joint resamples of attacks ({n_atk_sub}) AND benign ({n_ben_total}) for {r_name}...")

        sys_rec_list, sys_fpr_list, sys_prec_list = [], [], []
        t1_rec_list, t1_fpr_list, t1_prec_list = [], [], []
        t2_rec_list, t2_fpr_list, t2_prec_list = [], [], []

        for _ in range(1000):
            # Resample attacks with replacement (size = n_atk_sub)
            sub_atk_idx = rng_boot.choice(atk_indices, size=n_atk_sub, replace=True)
            # Resample benign with replacement (size = n_ben_total)
            sub_ben_idx = rng_boot.choice(ben_indices, size=n_ben_total, replace=True)

            # System metrics
            tp_sys = int(np.sum(sys_all[sub_atk_idx] == 1))
            fp_sys = int(np.sum(sys_all[sub_ben_idx] == 1))
            rec_sys = (tp_sys / n_atk_sub) * 100.0
            fpr_sys = (fp_sys / n_ben_total) * 100.0
            prec_sys = (tp_sys / (tp_sys + fp_sys) * 100.0) if (tp_sys + fp_sys) > 0 else 0.0

            sys_rec_list.append(rec_sys)
            sys_fpr_list.append(fpr_sys)
            sys_prec_list.append(prec_sys)

            # Tier 1 metrics
            tp_t1 = int(np.sum(pred1_all[sub_atk_idx] == 1))
            fp_t1 = int(np.sum(pred1_all[sub_ben_idx] == 1))
            rec_t1 = (tp_t1 / n_atk_sub) * 100.0
            fpr_t1 = (fp_t1 / n_ben_total) * 100.0
            prec_t1 = (tp_t1 / (tp_t1 + fp_t1) * 100.0) if (tp_t1 + fp_t1) > 0 else 0.0

            t1_rec_list.append(rec_t1)
            t1_fpr_list.append(fpr_t1)
            t1_prec_list.append(prec_t1)

            # Tier 2 metrics
            tp_t2 = int(np.sum(pred2_all[sub_atk_idx] == 1))
            fp_t2 = int(np.sum(pred2_all[sub_ben_idx] == 1))
            rec_t2 = (tp_t2 / n_atk_sub) * 100.0
            fpr_t2 = (fp_t2 / n_ben_total) * 100.0
            prec_t2 = (tp_t2 / (tp_t2 + fp_t2) * 100.0) if (tp_t2 + fp_t2) > 0 else 0.0

            t2_rec_list.append(rec_t2)
            t2_fpr_list.append(fpr_t2)
            t2_prec_list.append(prec_t2)

        # Full-sample point estimates and Wilson score intervals for FPR
        full_fp_sys = int(np.sum(sys_all[ben_indices] == 1))
        full_fp_t1 = int(np.sum(pred1_all[ben_indices] == 1))
        full_fp_t2 = int(np.sum(pred2_all[ben_indices] == 1))

        wilson_fpr_sys = compute_wilson_score_interval(full_fp_sys, n_ben_total)
        wilson_fpr_t1 = compute_wilson_score_interval(full_fp_t1, n_ben_total)
        wilson_fpr_t2 = compute_wilson_score_interval(full_fp_t2, n_ben_total)

        full_tp_sys = int(np.sum(sys_all[atk_indices] == 1))
        full_tp_t1 = int(np.sum(pred1_all[atk_indices] == 1))
        full_tp_t2 = int(np.sum(pred2_all[atk_indices] == 1))

        # Precision point estimate under prevalence ratio:
        # Expected TP = n_atk_sub * recall, Expected FP = n_ben_total * fpr
        rec_sys_pt = full_tp_sys / n_atk_total
        fpr_sys_pt = full_fp_sys / n_ben_total
        exp_tp_sys = n_atk_sub * rec_sys_pt
        exp_fp_sys = n_ben_total * fpr_sys_pt
        prec_sys_pt = (exp_tp_sys / (exp_tp_sys + exp_fp_sys) * 100.0) if (exp_tp_sys + exp_fp_sys) > 0 else 0.0

        rec_t1_pt = full_tp_t1 / n_atk_total
        fpr_t1_pt = full_fp_t1 / n_ben_total
        exp_tp_t1 = n_atk_sub * rec_t1_pt
        exp_fp_t1 = n_ben_total * fpr_t1_pt
        prec_t1_pt = (exp_tp_t1 / (exp_tp_t1 + exp_fp_t1) * 100.0) if (exp_tp_t1 + exp_fp_t1) > 0 else 0.0

        rec_t2_pt = full_tp_t2 / n_atk_total
        fpr_t2_pt = full_fp_t2 / n_ben_total
        exp_tp_t2 = n_atk_sub * rec_t2_pt
        exp_fp_t2 = n_ben_total * fpr_t2_pt
        prec_t2_pt = (exp_tp_t2 / (exp_tp_t2 + exp_fp_t2) * 100.0) if (exp_tp_t2 + exp_fp_t2) > 0 else 0.0

        prevalence_subsampling[r_name] = {
            "n_attacks_sampled": n_atk_sub,
            "n_benign_sampled": n_ben_total,
            "resampling_protocol": "Joint stratified bootstrap of attacks AND benign with replacement (1,000 resamples)",
            "sys": {
                "recall_point": float(rec_sys_pt * 100.0),
                "recall_ci": compute_percentile_ci(sys_rec_list),
                "fpr_point": float(fpr_sys_pt * 100.0),
                "fpr_wilson_ci": wilson_fpr_sys,
                "fpr_bootstrap_ci": compute_percentile_ci(sys_fpr_list),
                "precision_point": float(prec_sys_pt),
                "precision_ci": compute_percentile_ci(sys_prec_list),
            },
            "t1": {
                "recall_point": float(rec_t1_pt * 100.0),
                "recall_ci": compute_percentile_ci(t1_rec_list),
                "fpr_point": float(fpr_t1_pt * 100.0),
                "fpr_wilson_ci": wilson_fpr_t1,
                "fpr_bootstrap_ci": compute_percentile_ci(t1_fpr_list),
                "precision_point": float(prec_t1_pt),
                "precision_ci": compute_percentile_ci(t1_prec_list),
            },
            "t2": {
                "recall_point": float(rec_t2_pt * 100.0),
                "recall_ci": compute_percentile_ci(t2_rec_list),
                "fpr_point": float(fpr_t2_pt * 100.0),
                "fpr_wilson_ci": wilson_fpr_t2,
                "fpr_bootstrap_ci": compute_percentile_ci(t2_fpr_list),
                "precision_point": float(prec_t2_pt),
                "precision_ci": compute_percentile_ci(t2_prec_list),
            },
        }

    # ── 5. Assemble and Save Output ───────────────────────────────────────────
    final_output = {
        "execution_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "benchmark_source": UNIFORM_NPZ,
        "split_table_verified": verified_splits,
        "lofo_folds": lofo_folds_raw,
        "macro_summary": macro_summary,
        "window_ablation_74": {
            "folds": ablation_raw,
            "macro": macro_ablation,
        },
        "prevalence_resampling_1000": prevalence_subsampling,
        # Preserve backwards-compatible keys for report generators
        "prevalence_subsampling_1000": prevalence_subsampling,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[Uniform LOFO] Wrote uniform benchmark raw results to {OUT_JSON}")

    # Also update COMPAT_JSON so any existing downstream consumer reads uniform benchmark
    with open(COMPAT_JSON, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"[Uniform LOFO] Synced compatibility artifact {COMPAT_JSON}")


if __name__ == "__main__":
    run_uniform_lofo()
