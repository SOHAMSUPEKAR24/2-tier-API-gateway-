"""
scripts/compute_conformal_coverage_check.py
───────────────────────────────────────────
Computes Conformal Prediction Coverage Validity Check (Item 2).
Reports:
  - Nominal Alpha (e.g., 0.01, 0.05, 0.10)
  - Nominal Coverage (99%, 95%, 90%)
  - Empirical Prediction Set Coverage: P(Y in C(X))
  - Non-escalated flow fraction: P(|C(X)| == 1)
  - Error rate on non-escalated flows: P(Y != y_hat | |C(X)| == 1) vs alpha
  - Escalated flow fraction (Ambiguity rate): P(|C(X)| != 1)
Evaluated across 5 seeds [42, 71, 137, 256, 512] with 95% confidence intervals.
"""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.result_writer import mean_ci95

BENCHMARK_NPZ = "data/processed/benchmark_dataset.npz"
SEEDS = [42, 71, 137, 256, 512]
ALPHAS = [0.01, 0.05, 0.10]


def run_conformal_check():
    bench = np.load(BENCHMARK_NPZ)
    X_tr = bench["ftp_only_X_train"]
    y_tr = bench["ftp_only_y_train"]
    X_te = bench["ftp_only_X_test"]
    y_te = bench["ftp_only_y_test"]

    results_by_alpha = {a: [] for a in ALPHAS}

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        # Split training set into fit (75%) and calibration (25%)
        cal_idx = rng.choice(len(X_tr), size=int(0.25 * len(X_tr)), replace=False)
        fit_idx = np.setdiff1d(np.arange(len(X_tr)), cal_idx)

        X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]
        X_cal, y_cal = X_tr[cal_idx], y_tr[cal_idx]

        rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed, n_jobs=-1)
        rf.fit(X_fit, y_fit)

        # Calibrate non-conformity scores s_i = 1 - P(Y = y_i | X_i)
        p_cal = rf.predict_proba(X_cal)
        cal_scores = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]

        p_te = rf.predict_proba(X_te)
        n_cal = len(cal_scores)

        for alpha in ALPHAS:
            q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
            q_val = float(np.quantile(cal_scores, min(1.0, q_level)))

            # Construct prediction sets C(X)
            # y in C(X) iff 1 - P(Y=y|X) <= q_val  <=>  P(Y=y|X) >= 1 - q_val
            c_0 = (1.0 - p_te[:, 0]) <= q_val
            c_1 = (1.0 - p_te[:, 1]) <= q_val

            # Overall coverage P(Y in C(X))
            in_set = np.where(y_te == 0, c_0, c_1)
            empirical_set_cov = float(np.mean(in_set) * 100.0)

            # Singleton prediction sets (non-escalated)
            singleton_mask = (c_0 ^ c_1)  # exactly one class is included
            singleton_share = float(np.mean(singleton_mask) * 100.0)

            # Escalated sets (|C(X)| != 1: either {0,1} or empty)
            esc_share = float(np.mean(~singleton_mask) * 100.0)

            # Error rate on non-escalated flows
            y_hat_singleton = np.where(c_1[singleton_mask], 1, 0)
            y_true_singleton = y_te[singleton_mask]
            err_non_esc = float(np.mean(y_hat_singleton != y_true_singleton) * 100.0)

            results_by_alpha[alpha].append({
                "set_coverage": empirical_set_cov,
                "singleton_share": singleton_share,
                "escalated_share": esc_share,
                "err_non_esc": err_non_esc,
            })

    summary = {}
    for alpha in ALPHAS:
        recs = results_by_alpha[alpha]
        sc_m, sc_ci = mean_ci95([r["set_coverage"] for r in recs])
        ss_m, ss_ci = mean_ci95([r["singleton_share"] for r in recs])
        es_m, es_ci = mean_ci95([r["escalated_share"] for r in recs])
        er_m, er_ci = mean_ci95([r["err_non_esc"] for r in recs])

        summary[f"alpha_{alpha}"] = {
            "nominal_alpha": alpha,
            "nominal_coverage_pct": (1.0 - alpha) * 100.0,
            "set_coverage_mean": sc_m,
            "set_coverage_ci95": sc_ci,
            "non_escalated_share_mean": ss_m,
            "non_escalated_share_ci95": ss_ci,
            "escalated_share_mean": es_m,
            "escalated_share_ci95": es_ci,
            "err_on_non_escalated_mean": er_m,
            "err_on_non_escalated_ci95": er_ci,
        }

    out_file = Path("results/escalation_comparison/conformal_coverage_check.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_conformal_check()
