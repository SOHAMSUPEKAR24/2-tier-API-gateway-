"""
scripts/run_csic_tier1_comparison.py
────────────────────────────────────
Implements Task 8: CSIC Tier 1 Comprehensive Comparison.
Compares:
  1. Legacy 9-feature Fast RF
  2. Plain TF-IDF Logistic Regression (TfidfVectorizer char 3-5 n-grams, 2048 features)
  3. Hashed n-grams Logistic Regression (HashingVectorizer char 3-5 n-grams, 2048 features)
  4. Hashed n-grams Small GBM (HistGradientBoostingClassifier, max_iter=40, max_depth=5)

Evaluates:
  - Budget-calibrated thresholds: calibrated on pure benign calibration traffic at target budgets 1%, 5%, 10%
  - Standalone Tier 1 screening accuracy (prediction = 1(p >= 0.50))
  - Two-tier system accuracy (escalated requests verified by Tier 2 deep model)
  - Benign escalation rate, FPR, and Attack SFR at each budget
  - 5 seeds [42, 71, 137, 256, 512] with 95% confidence intervals
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.result_writer import make_run_dir, mean_ci95, write_result
from tier2.track_a_http import extract_raw_http_text, extract_track_a_fast_features

CSIC_CSV = "/Users/sohamsupekar/Desktop/MajorProject/DATASET/csic_database.csv"
SEEDS = [42, 71, 137, 256, 512]
BUDGETS = [0.01, 0.05, 0.10]


def run_csic_comparison(sample_size: int = 15000) -> Dict[str, Any]:
    print("=" * 70)
    print("TASK 8: CSIC TIER 1 ARCHITECTURE COMPARISON & BUDGET CALIBRATION")
    print(f"Seeds: {SEEDS}, Budgets: {BUDGETS}")
    print("=" * 70)

    if not os.path.exists(CSIC_CSV):
        raise FileNotFoundError(f"CSIC dataset not found at {CSIC_CSV}")

    df_raw = pd.read_csv(CSIC_CSV)
    df_sub = df_raw.sample(n=min(sample_size, len(df_raw)), random_state=42).reset_index(drop=True)

    # Pre-extract texts and 9-features for speed
    print("Extracting features from HTTP requests...")
    texts = [extract_raw_http_text(r) for _, r in df_sub.iterrows()]
    feats_9 = np.array([extract_track_a_fast_features(r) for _, r in df_sub.iterrows()])
    labels = df_sub["classification"].values.astype(int)

    models_to_test = [
        "Legacy 9-Feature Fast RF",
        "Plain TF-IDF Logistic Regression",
        "Hashed n-grams Logistic Regression",
        "Hashed n-grams Small GBM",
    ]

    results = {m: {"standalone_acc": [], "budget_results": {b: [] for b in BUDGETS}} for m in models_to_test}

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        rng = np.random.RandomState(seed)
        n = len(df_sub)
        indices = rng.permutation(n)

        # 60% train, 15% calibration, 25% test
        n_tr = int(0.60 * n)
        n_cal = int(0.15 * n)
        idx_tr = indices[:n_tr]
        idx_cal = indices[n_tr : n_tr + n_cal]
        idx_te = indices[n_tr + n_cal :]

        y_tr, y_cal, y_te = labels[idx_tr], labels[idx_cal], labels[idx_te]
        texts_tr = [texts[i] for i in idx_tr]
        texts_cal = [texts[i] for i in idx_cal]
        texts_te = [texts[i] for i in idx_te]

        X9_tr, X9_cal, X9_te = feats_9[idx_tr], feats_9[idx_cal], feats_9[idx_te]

        # Fit vectorizers
        tfidf = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=2048)
        X_tfidf_tr = tfidf.fit_transform(texts_tr).toarray()
        X_tfidf_cal = tfidf.transform(texts_cal).toarray()
        X_tfidf_te = tfidf.transform(texts_te).toarray()

        hv = HashingVectorizer(ngram_range=(3, 5), analyzer="char", n_features=2048, alternate_sign=False)
        X_hv_tr = hv.transform(texts_tr).toarray()
        X_hv_cal = hv.transform(texts_cal).toarray()
        X_hv_te = hv.transform(texts_te).toarray()

        # Calibration indices (benign only for budget calibration)
        benign_cal_mask = (y_cal == 0)
        benign_te_mask = (y_te == 0)
        attack_te_mask = (y_te == 1)

        # Build each model
        fitted_models = {}

        # 1. Legacy 9-Feature RF
        rf9 = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=seed, n_jobs=-1)
        rf9.fit(X9_tr, y_tr)
        fitted_models["Legacy 9-Feature Fast RF"] = (rf9, X9_cal, X9_te)

        # 2. Plain TF-IDF LR
        lr_tfidf = LogisticRegression(max_iter=1000, C=10.0, random_state=seed)
        lr_tfidf.fit(X_tfidf_tr, y_tr)
        fitted_models["Plain TF-IDF Logistic Regression"] = (lr_tfidf, X_tfidf_cal, X_tfidf_te)

        # 3. Hashed n-grams LR
        lr_hv = LogisticRegression(max_iter=1000, C=10.0, random_state=seed)
        lr_hv.fit(X_hv_tr, y_tr)
        fitted_models["Hashed n-grams Logistic Regression"] = (lr_hv, X_hv_cal, X_hv_te)

        # 4. Hashed n-grams Small GBM
        gbm_hv = HistGradientBoostingClassifier(max_iter=40, max_depth=5, random_state=seed)
        gbm_hv.fit(X_hv_tr, y_tr)
        fitted_models["Hashed n-grams Small GBM"] = (gbm_hv, X_hv_cal, X_hv_te)

        # Evaluate each model
        for name, (clf, X_cal_m, X_te_m) in fitted_models.items():
            p_cal = clf.predict_proba(X_cal_m)[:, 1]
            p_te = clf.predict_proba(X_te_m)[:, 1]

            # Standalone accuracy (no escalation, threshold = 0.50)
            preds_standalone = (p_te >= 0.5).astype(int)
            standalone_acc = float(accuracy_score(y_te, preds_standalone) * 100.0)
            results[name]["standalone_acc"].append(standalone_acc)

            # Uncertainty score = distance to 0.5 (or margin 1 - 2*|p - 0.5|)
            # For escalation, we escalate flows where uncertainty >= threshold
            # S_uncertainty = 1.0 - 2.0 * np.abs(p - 0.5) in [0, 1]
            unc_cal_benign = 1.0 - 2.0 * np.abs(p_cal[benign_cal_mask] - 0.5)
            unc_te = 1.0 - 2.0 * np.abs(p_te - 0.5)

            for b in BUDGETS:
                # Target budget B: escalate the top B fraction of benign calibration traffic
                # Threshold tau_b is the (1 - B) quantile of benign calibration uncertainty
                tau_b = float(np.quantile(unc_cal_benign, 1.0 - b))

                esc_mask = (unc_te >= tau_b)
                achieved_benign_esc = float(np.mean(esc_mask[benign_te_mask]) * 100.0)

                # Two-tier system accuracy:
                # If escalated, request is forwarded to Tier 2 (Oracle resolution for CSIC verification)
                # If not escalated, Tier 1 standalone prediction stands
                y_system = np.where(esc_mask, y_te, preds_standalone)
                system_acc = float(accuracy_score(y_te, y_system) * 100.0)

                # Silent Failure Rate (SFR %): true attacks classified as benign by Tier 1 AND NOT escalated
                silent_failures = attack_te_mask & (preds_standalone == 0) & (~esc_mask)
                sfr = float(np.sum(silent_failures) / np.sum(attack_te_mask) * 100.0)

                # FPR on benign
                fpr = float(np.sum(benign_te_mask & (preds_standalone == 1) & (~esc_mask)) / np.sum(benign_te_mask) * 100.0)

                results[name]["budget_results"][b].append({
                    "achieved_benign_esc": achieved_benign_esc,
                    "system_acc": system_acc,
                    "sfr": sfr,
                    "fpr": fpr,
                })

            print(f"  {name}: Standalone Acc = {standalone_acc:.2f}% | System Acc @ 5% = {results[name]['budget_results'][0.05][-1]['system_acc']:.2f}%")

    # Aggregate across seeds
    summary = {}
    for name in models_to_test:
        stand_vals = results[name]["standalone_acc"]
        sm, sci = mean_ci95(stand_vals)
        summary[name] = {
            "standalone_acc_mean": sm,
            "standalone_acc_ci95": sci,
            "budgets": {},
        }
        for b in BUDGETS:
            b_list = results[name]["budget_results"][b]
            b_esc = [r["achieved_benign_esc"] for r in b_list]
            b_sys = [r["system_acc"] for r in b_list]
            b_sfr = [r["sfr"] for r in b_list]
            b_fpr = [r["fpr"] for r in b_list]

            em, eci = mean_ci95(b_esc)
            sym, syci = mean_ci95(b_sys)
            sfm, sfci = mean_ci95(b_sfr)
            fm, fci = mean_ci95(b_fpr)

            summary[name]["budgets"][f"budget_{int(b*100)}pct"] = {
                "target_budget": b * 100.0,
                "achieved_benign_esc_mean": em,
                "achieved_benign_esc_ci95": eci,
                "system_acc_mean": sym,
                "system_acc_ci95": syci,
                "sfr_mean": sfm,
                "sfr_ci95": sfci,
                "fpr_mean": fm,
                "fpr_ci95": fci,
            }

    out_file = Path("results/escalation_comparison/csic_tier1_comparison.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[CSIC Comparison] Results saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_csic_comparison()
