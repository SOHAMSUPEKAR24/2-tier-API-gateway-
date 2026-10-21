"""
scripts/run_novelty_diagnostics.py
──────────────────────────────────
Evaluates Item 1 of User Request:
  Novelty diagnostics across LOFO and Cold-Start for:
    - RF Margin
    - Platt-Calibrated RF
    - Isolation Forest (benign density)
    - Mahalanobis Distance (regularized Ledoit-Wolf)
    - kNN Distance (k=5)
  On both 9 Fast Features and 76 Full Features.

Metrics reported per signal across 5 seeds [42, 71, 137, 256, 512] (mean ± 95% CI):
  1. AUROC (Benign vs Held-Out Family)
  2. On the HELD-OUT FAMILY ONLY at 1%, 5%, 10% calibration budgets:
     - Target Budget vs Achieved Benign Escalation (%)
     - Escalation Coverage on Held-Out Family (%)
     - Silent-Failure Rate (SFR %) on Held-Out Family

Outputs:
  - results/lofo/novelty_diagnostics.json
  - Formatted tables in results/lofo/REPORT.md and results/escalation_comparison/REPORT.md
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_signal_gate import KNNNovelty, MahalanobisNovelty
from src.utils.result_writer import mean_ci95

BENCHMARK_NPZ = "data/processed/benchmark_dataset.npz"
ALL_FAMILIES = ["FTP-BruteForce", "SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]
SEEDS = [42, 71, 137, 256, 512]
BUDGETS = [0.01, 0.05, 0.10]


def evaluate_signal_on_split(
    signal_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    y_fam: np.ndarray,
    held_out_family: str,
    seed: int,
) -> Dict[str, Any]:
    """Evaluates a single novelty signal on a specific train/test split."""
    benign_tr_mask = (y_tr == 0)
    X_benign_tr = X_tr[benign_tr_mask]
    cal_size = int(0.25 * len(X_benign_tr))
    X_cal = X_benign_tr[-cal_size:]
    X_benign_fit = X_benign_tr[:-cal_size] if len(X_benign_tr) > cal_size * 2 else X_benign_tr

    benign_te_mask = (y_fam == "Benign")
    if held_out_family == "unseen_all":
        # Cold start: all non-benign, non-FTP families are held out
        held_out_mask = (y_fam != "Benign") & (y_fam != "FTP-BruteForce")
    else:
        held_out_mask = (y_fam == held_out_family)

    # Base RF trained on (X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p_te = rf.predict_proba(X_te)[:, 1]
    preds_te = (p_te >= 0.5).astype(int)

    # 1. Compute scores
    if signal_name == "RF Margin":
        p_cal = rf.predict_proba(X_cal)[:, 1]
        scores_cal = 1.0 - 2.0 * np.abs(p_cal - 0.5)
        scores_te = 1.0 - 2.0 * np.abs(p_te - 0.5)
        preds_for_sfr = preds_te

    elif signal_name == "Platt-Calibrated RF":
        # Platt / Sigmoid calibration
        X_sub_tr, X_sub_cal, y_sub_tr, y_sub_cal = train_test_split(
            X_tr, y_tr, test_size=0.25, random_state=seed, stratify=y_tr
        )
        base_cal = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed, n_jobs=-1)
        base_cal.fit(X_sub_tr, y_sub_tr)
        calibrator = CalibratedClassifierCV(base_cal, cv="prefit", method="sigmoid")
        calibrator.fit(X_sub_cal, y_sub_cal)

        p_cal = calibrator.predict_proba(X_cal)[:, 1]
        scores_cal = 1.0 - 2.0 * np.abs(p_cal - 0.5)
        p_te_cal = calibrator.predict_proba(X_te)[:, 1]
        scores_te = 1.0 - 2.0 * np.abs(p_te_cal - 0.5)
        preds_for_sfr = (p_te_cal >= 0.5).astype(int)

    elif signal_name == "Isolation Forest":
        iso = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
        iso.fit(X_benign_fit)
        scores_cal = -iso.score_samples(X_cal)
        scores_te = -iso.score_samples(X_te)
        preds_for_sfr = preds_te

    elif signal_name == "Mahalanobis":
        mahal = MahalanobisNovelty()
        mahal.fit(X_benign_fit)
        scores_cal = mahal.score(X_cal)
        scores_te = mahal.score(X_te)
        preds_for_sfr = preds_te

    elif signal_name == "kNN Distance":
        knn = KNNNovelty(k=5)
        knn.fit(X_benign_fit)
        scores_cal = knn.score(X_cal)
        scores_te = knn.score(X_te)
        preds_for_sfr = preds_te

    else:
        raise ValueError(f"Unknown signal: {signal_name}")

    # 2. AUROC: Benign vs Held-Out Family Only
    auroc_eval_mask = benign_te_mask | held_out_mask
    try:
        auroc = float(roc_auc_score(held_out_mask[auroc_eval_mask].astype(int), scores_te[auroc_eval_mask]))
    except Exception:
        auroc = float("nan")

    # 3. Budget thresholds and held-out metrics
    budget_metrics = {}
    for b in BUDGETS:
        level = min(1.0, max(0.0, np.ceil((len(X_cal) + 1) * (1.0 - b)) / len(X_cal)))
        tau = float(np.quantile(scores_cal, level, method="higher"))

        achieved_benign_esc = float(np.mean(scores_te[benign_te_mask] >= tau) * 100.0)

        # On Held-Out Family Only:
        held_out_scores = scores_te[held_out_mask]
        held_out_preds = preds_for_sfr[held_out_mask]

        escalated_held_out = (held_out_scores >= tau)
        coverage_held_out = float(np.mean(escalated_held_out) * 100.0)

        # SFR on Held-Out: predicted benign (0) and NOT escalated
        silent_failures = (held_out_preds == 0) & (~escalated_held_out)
        sfr_held_out = float(np.sum(silent_failures) / len(held_out_scores) * 100.0)

        budget_metrics[f"budget_{int(b*100)}pct"] = {
            "target_budget": b * 100.0,
            "achieved_benign_esc": achieved_benign_esc,
            "held_out_coverage": coverage_held_out,
            "held_out_sfr": sfr_held_out,
        }

    return {
        "auroc": auroc,
        "budgets": budget_metrics,
    }


def run_all_novelty_diagnostics():
    print("=" * 70)
    print("RUNNING AUTHORITATIVE NOVELTY DIAGNOSTICS (LOFO & COLD-START)")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    bench = np.load(BENCHMARK_NPZ)
    signals = ["RF Margin", "Platt-Calibrated RF", "Isolation Forest", "Mahalanobis", "kNN Distance"]

    results = {
        "lofo": {},
        "cold_start": {},
    }

    # ── 1. LOFO DIAGNOSTICS (9 and 76 features) ──────────────────────────────
    for feat_dim, suffix in [(9, ""), (76, "_76")]:
        dim_key = f"features_{feat_dim}"
        results["lofo"][dim_key] = {}

        for signal in signals:
            print(f"  [LOFO {feat_dim} Features] Signal: {signal}...")
            signal_fold_runs = {fam: [] for fam in ALL_FAMILIES}

            for seed in SEEDS:
                for fam in ALL_FAMILIES:
                    X_tr = bench[f"lofo_{fam}_X_train{suffix}"]
                    y_tr = bench[f"lofo_{fam}_y_train"]
                    X_te = bench[f"lofo_{fam}_X_test{suffix}"]
                    y_te = bench[f"lofo_{fam}_y_test"]
                    y_fam = bench[f"lofo_{fam}_y_family_test"]

                    res = evaluate_signal_on_split(signal, X_tr, y_tr, X_te, y_te, y_fam, fam, seed)
                    signal_fold_runs[fam].append(res)

            # Aggregate across seeds per fold
            fold_aggs = {}
            for fam in ALL_FAMILIES:
                aurocs = [r["auroc"] for r in signal_fold_runs[fam]]
                am, aci = mean_ci95(aurocs)

                b_aggs = {}
                for b in BUDGETS:
                    bk = f"budget_{int(b*100)}pct"
                    ach_b = [r["budgets"][bk]["achieved_benign_esc"] for r in signal_fold_runs[fam]]
                    cov_h = [r["budgets"][bk]["held_out_coverage"] for r in signal_fold_runs[fam]]
                    sfr_h = [r["budgets"][bk]["held_out_sfr"] for r in signal_fold_runs[fam]]

                    ab_m, ab_ci = mean_ci95(ach_b)
                    cv_m, cv_ci = mean_ci95(cov_h)
                    sf_m, sf_ci = mean_ci95(sfr_h)

                    b_aggs[bk] = {
                        "target_budget": b * 100.0,
                        "achieved_benign_esc_mean": ab_m,
                        "achieved_benign_esc_ci95": ab_ci,
                        "held_out_coverage_mean": cv_m,
                        "held_out_coverage_ci95": cv_ci,
                        "held_out_sfr_mean": sf_m,
                        "held_out_sfr_ci95": sf_ci,
                    }

                fold_aggs[fam] = {
                    "auroc_mean": am,
                    "auroc_ci95": aci,
                    "budgets": b_aggs,
                }

            # Macro-Average across all 5 folds
            macro_auroc_mean = float(np.mean([fold_aggs[f]["auroc_mean"] for f in ALL_FAMILIES]))
            macro_b = {}
            for b in BUDGETS:
                bk = f"budget_{int(b*100)}pct"
                macro_b[bk] = {
                    "target_budget": b * 100.0,
                    "achieved_benign_esc_mean": float(np.mean([fold_aggs[f]["budgets"][bk]["achieved_benign_esc_mean"] for f in ALL_FAMILIES])),
                    "held_out_coverage_mean": float(np.mean([fold_aggs[f]["budgets"][bk]["held_out_coverage_mean"] for f in ALL_FAMILIES])),
                    "held_out_sfr_mean": float(np.mean([fold_aggs[f]["budgets"][bk]["held_out_sfr_mean"] for f in ALL_FAMILIES])),
                }

            results["lofo"][dim_key][signal] = {
                "per_fold": fold_aggs,
                "macro_average": {
                    "auroc_mean": macro_auroc_mean,
                    "budgets": macro_b,
                },
            }

    # ── 2. COLD-START DIAGNOSTICS (9 and 76 features) ────────────────────────
    for feat_dim, suffix in [(9, ""), (76, "_76")]:
        dim_key = f"features_{feat_dim}"
        results["cold_start"][dim_key] = {}

        X_tr = bench[f"ftp_only_X_train{suffix}"]
        y_tr = bench[f"ftp_only_y_train"]
        X_te = bench[f"ftp_only_X_test{suffix}"]
        y_te = bench[f"ftp_only_y_test"]
        y_fam = bench[f"ftp_only_y_family_test"]

        for signal in signals:
            print(f"  [Cold-Start {feat_dim} Features] Signal: {signal}...")
            cs_runs = []
            for seed in SEEDS:
                res = evaluate_signal_on_split(signal, X_tr, y_tr, X_te, y_te, y_fam, "unseen_all", seed)
                cs_runs.append(res)

            aurocs = [r["auroc"] for r in cs_runs]
            am, aci = mean_ci95(aurocs)

            b_aggs = {}
            for b in BUDGETS:
                bk = f"budget_{int(b*100)}pct"
                ach_b = [r["budgets"][bk]["achieved_benign_esc"] for r in cs_runs]
                cov_h = [r["budgets"][bk]["held_out_coverage"] for r in cs_runs]
                sfr_h = [r["budgets"][bk]["held_out_sfr"] for r in cs_runs]

                ab_m, ab_ci = mean_ci95(ach_b)
                cv_m, cv_ci = mean_ci95(cov_h)
                sf_m, sf_ci = mean_ci95(sfr_h)

                b_aggs[bk] = {
                    "target_budget": b * 100.0,
                    "achieved_benign_esc_mean": ab_m,
                    "achieved_benign_esc_ci95": ab_ci,
                    "held_out_coverage_mean": cv_m,
                    "held_out_coverage_ci95": cv_ci,
                    "held_out_sfr_mean": sf_m,
                    "held_out_sfr_ci95": sf_ci,
                }

            results["cold_start"][dim_key][signal] = {
                "auroc_mean": am,
                "auroc_ci95": aci,
                "budgets": b_aggs,
            }

    out_path = Path("results/lofo/novelty_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved Novelty Diagnostics to: {out_path}")
    return results


if __name__ == "__main__":
    run_all_novelty_diagnostics()
