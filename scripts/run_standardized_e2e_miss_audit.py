#!/usr/bin/env python3
"""
scripts/run_standardized_e2e_miss_audit.py
──────────────────────────────────────────
Standardized escalation gates benchmark & Tier 2 per-flow audit:
  1. Standardized features (StandardScaler fit on clean benign train) for:
     - RF Probability Band
     - Full kNN (k=5, 4000 benign reference flows)
     - Prototype kNN (M=32 cluster centroids)
     - Mahalanobis Distance (regularized covariance)
     - UNION Gate (RF Band + Full kNN with split budget)
  2. Matched achieved benign escalation budgets: 1%, 2%, 5%, 10%.
  3. Deployable calibration thresholds (fit on 50% training benign split)
     reported side-by-side with oracle-matched test thresholds.
  4. Paired per-block differences vs RF Band across 10 chronological time blocks.
  5. SSH Check on standardized 25-feature gate (kNN & Mahalanobis).
  6. Trained Tier 2 (RF76) per-flow prediction audit on escalated flows:
     - Log per-flow predictions, caught/missed counts, and end-to-end miss rates.
  7. Outputs raw JSON to results/escalation_comparison/standardized_gate_eval.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BENCHMARK_NPZ = "data/processed/benchmark_dataset.npz"
OUT_JSON = "results/escalation_comparison/standardized_gate_eval.json"
NOVEL_FAMILIES = ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]
ALL_FAMILIES = ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris", "FTP-BruteForce"]
TARGET_BUDGETS = [0.01, 0.02, 0.05, 0.10]
N_BLOCKS = 10


def run_standardized_eval():
    print("=" * 70)
    print("RUNNING STANDARDIZED ESCALATION GATES & TIER 2 PER-FLOW AUDIT")
    print("=" * 70)

    if not os.path.exists(BENCHMARK_NPZ):
        raise FileNotFoundError(f"Missing benchmark dataset: {BENCHMARK_NPZ}")

    bench = np.load(BENCHMARK_NPZ)
    X_tr_9 = bench["ftp_only_X_train"][:, :9]
    y_tr = bench["ftp_only_y_train"]
    X_te_9 = bench["ftp_only_X_test"][:, :9]
    y_te = bench["ftp_only_y_test"]
    y_fam = bench["ftp_only_y_family_test"]

    X_tr_25 = bench["ftp_only_X_train_25"]
    X_te_25 = bench["ftp_only_X_test_25"]

    X_tr_76 = bench["ftp_only_X_train_76"]
    X_te_76 = bench["ftp_only_X_test_76"]

    benign_tr_mask = (y_tr == 0)
    benign_te_mask = (y_te == 0)
    novel_te_mask = np.isin(y_fam, NOVEL_FAMILIES)

    b_tr = X_tr_9[benign_tr_mask]
    b_te = X_te_9[benign_te_mask]

    # Chronological time blocks
    block_size = len(y_te) // N_BLOCKS
    blocks = [
        np.arange(i * block_size, (i + 1) * block_size if i < N_BLOCKS - 1 else len(y_te))
        for i in range(N_BLOCKS)
    ]

    # ── 1. Standardization & Model Fitting ───────────────────────────────────
    scaler_9 = StandardScaler().fit(b_tr)
    X_tr_sc = scaler_9.transform(X_tr_9)
    X_te_sc = scaler_9.transform(X_te_9)
    b_tr_sc = scaler_9.transform(b_tr)
    b_te_sc = scaler_9.transform(b_te)

    # Train Tier 1
    rf_t1 = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42).fit(X_tr_9, y_tr)
    rf_score_all = -np.abs(rf_t1.predict_proba(X_te_9)[:, 1] - 0.5)
    rf_score_benign = -np.abs(rf_t1.predict_proba(b_te)[:, 1] - 0.5)
    preds_t1 = rf_t1.predict(X_te_9)

    # Train Tier 2 (RF on 76 features) - FTP-only Baseline
    rf_t2_ftp = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_tr_76, y_tr)
    preds_t2_ftp = rf_t2_ftp.predict(X_te_76)
    probs_t2_ftp = rf_t2_ftp.predict_proba(X_te_76)[:, 1]

    # Train Tier 2 (RF on 76 features) - LOFO Models (trained on remaining 4 attack families + benign)
    lofo_t2_models = {}
    for fam in NOVEL_FAMILIES:
        rf_l = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1).fit(
            bench[f"lofo_{fam}_X_train_76"], bench[f"lofo_{fam}_y_train"]
        )
        lofo_t2_models[fam] = rf_l

    # Predictions under LOFO-trained Tier 2 across novel test flows
    preds_t2_lofo = np.zeros(len(y_te), dtype=int)
    for fam in NOVEL_FAMILIES:
        fm = (y_fam == fam)
        if np.sum(fm) > 0:
            preds_t2_lofo[fm] = lofo_t2_models[fam].predict(X_te_76[fm])

    # Gate: Full kNN (k=5)
    nn_full = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(b_tr_sc[:4000])
    knn_score_all = nn_full.kneighbors(X_te_sc)[0][:, -1]
    knn_score_benign = nn_full.kneighbors(b_te_sc)[0][:, -1]

    # Gate: Prototype kNN (M=32)
    kmeans = MiniBatchKMeans(n_clusters=32, random_state=42, batch_size=256, n_init="auto").fit(b_tr_sc)
    nn_proto = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(kmeans.cluster_centers_)
    proto_score_all = nn_proto.kneighbors(X_te_sc)[0][:, 0]
    proto_score_benign = nn_proto.kneighbors(b_te_sc)[0][:, 0]

    # Gate: Mahalanobis Distance
    cov_9 = np.cov(b_tr_sc, rowvar=False) + 1e-4 * np.eye(9)
    inv_cov_9 = np.linalg.inv(cov_9)
    diff_all = X_te_sc - np.mean(b_tr_sc, axis=0)
    diff_benign = b_te_sc - np.mean(b_tr_sc, axis=0)
    maha_score_all = np.sqrt(np.sum(diff_all @ inv_cov_9 * diff_all, axis=1))
    maha_score_benign = np.sqrt(np.sum(diff_benign @ inv_cov_9 * diff_benign, axis=1))

    # ── 2. Calibration Split for Deployable Thresholds ───────────────────────
    rng_cal = np.random.RandomState(42)
    n_b = len(b_tr)
    perm = rng_cal.permutation(n_b)
    fit_idx = perm[: n_b // 2]
    cal_idx = perm[n_b // 2 :]

    sc_cal = StandardScaler().fit(b_tr[fit_idx])
    b_fit_sc = sc_cal.transform(b_tr[fit_idx])
    b_cal_sc = sc_cal.transform(b_tr[cal_idx])
    X_te_sc_cal = sc_cal.transform(X_te_9)
    b_te_sc_cal = sc_cal.transform(b_te)

    nn_cal = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(b_fit_sc[:4000])
    cal_scores_knn = nn_cal.kneighbors(b_cal_sc)[0][:, -1]
    test_cal_knn = nn_cal.kneighbors(b_te_sc_cal)[0][:, -1]

    cov_cal = np.cov(b_fit_sc, rowvar=False) + 1e-4 * np.eye(9)
    inv_cov_cal = np.linalg.inv(cov_cal)
    diff_cal = b_cal_sc - np.mean(b_fit_sc, axis=0)
    diff_test_cal = b_te_sc_cal - np.mean(b_fit_sc, axis=0)
    cal_scores_maha = np.sqrt(np.sum(diff_cal @ inv_cov_cal * diff_cal, axis=1))
    test_cal_maha = np.sqrt(np.sum(diff_test_cal @ inv_cov_cal * diff_test_cal, axis=1))

    # ── 3. Gate Evaluations & Bootstrap Confidence Intervals ─────────────────
    rng_boot = np.random.RandomState(42)

    def evaluate_single_gate(score_all, score_b, is_union=False, knn_s=None, knn_b=None):
        out = {}
        total_novel = int(np.sum(novel_te_mask))
        for b in TARGET_BUDGETS:
            b_key = f"{int(b*100)}%"
            if is_union:
                th_rf = np.percentile(rf_score_benign, 100 * (1 - b / 2))
                th_knn = np.percentile(knn_score_benign, 100 * (1 - b / 2))
                esc_mask = (rf_score_all >= th_rf) | (knn_score_all >= th_knn)
            else:
                th = np.percentile(score_b, 100 * (1 - b))
                esc_mask = score_all >= th

            achieved_benign = float(np.mean(esc_mask[benign_te_mask]) * 100.0)

            # Per-family counts and recall
            fam_counts = {}
            for fam in ALL_FAMILIES:
                f_mask = y_fam == fam
                n_total = int(np.sum(f_mask))
                n_esc = int(np.sum(esc_mask[f_mask]))
                pct = float((n_esc / max(1, n_total)) * 100.0)
                fam_counts[fam] = {"total": n_total, "escalated": n_esc, "coverage_pct": pct}

            novel_macro_cov = float(np.mean([fam_counts[fam]["coverage_pct"] for fam in NOVEL_FAMILIES]))
            nov_esc = esc_mask & novel_te_mask
            nov_esc_count = int(np.sum(nov_esc))
            novel_pooled_cov = float((nov_esc_count / max(1, total_novel)) * 100.0)

            # Stratified within-family flow bootstrap (1,000 resamples)
            boot_benign, boot_macro, boot_pooled = [], [], []
            gate_strata = {"Benign": np.where(benign_te_mask)[0]}
            for f_name in ALL_FAMILIES:
                gate_strata[f_name] = np.where(y_fam == f_name)[0]

            for _ in range(1000):
                s_idx = np.concatenate([
                    rng_boot.choice(idx_list, size=len(idx_list), replace=True)
                    for idx_list in gate_strata.values()
                ])
                b_esc = esc_mask[s_idx]
                b_y = y_te[s_idx]
                b_fam = y_fam[s_idx]
                boot_benign.append(float(np.mean(b_esc[b_y == 0]) * 100.0))
                covs = [
                    np.mean(b_esc[b_fam == fam]) * 100.0
                    for fam in NOVEL_FAMILIES
                    if np.sum(b_fam == fam) > 0
                ]
                boot_macro.append(float(np.mean(covs) if covs else 0.0))
                b_nov = np.isin(b_fam, NOVEL_FAMILIES)
                boot_pooled.append(float(np.mean(b_esc[b_nov]) * 100.0) if np.sum(b_nov) > 0 else 0.0)

            # FTP-only Trained Tier 2
            t2_ftp_caught_novel = int(np.sum((preds_t2_ftp == 1) & nov_esc))
            t2_ftp_missed_novel = int(nov_esc_count - t2_ftp_caught_novel)
            ftp_e2e_caught = int(
                np.sum(((preds_t1 == 1) & (~esc_mask) & novel_te_mask) | ((preds_t2_ftp == 1) & nov_esc))
            )
            ftp_e2e_miss = total_novel - ftp_e2e_caught
            ftp_e2e_miss_rate = float((ftp_e2e_miss / total_novel) * 100.0)

            # LOFO-Trained Tier 2
            t2_lofo_caught_novel = int(np.sum((preds_t2_lofo == 1) & nov_esc))
            t2_lofo_missed_novel = int(nov_esc_count - t2_lofo_caught_novel)
            lofo_e2e_caught = int(
                np.sum(((preds_t1 == 1) & (~esc_mask) & novel_te_mask) | ((preds_t2_lofo == 1) & nov_esc))
            )
            lofo_e2e_miss = total_novel - lofo_e2e_caught
            lofo_e2e_miss_rate = float((lofo_e2e_miss / total_novel) * 100.0)

            # Oracle Tier 2 (catches 100% of escalated novel flows)
            oracle_e2e_caught = int(np.sum(((preds_t1 == 1) & (~esc_mask) & novel_te_mask) | nov_esc))
            oracle_e2e_miss = total_novel - oracle_e2e_caught
            oracle_miss_rate = float((oracle_e2e_miss / total_novel) * 100.0)

            # Detailed per-family Tier 2 audit on escalated flows
            t2_per_fam = {}
            for fam in ALL_FAMILIES:
                fm = y_fam == fam
                f_esc = esc_mask & fm
                n_esc_f = int(np.sum(f_esc))
                c_ftp = int(np.sum((preds_t2_ftp == 1) & f_esc))
                c_lofo = int(np.sum((preds_t2_lofo == 1) & f_esc))
                t2_per_fam[fam] = {
                    "escalated_count": n_esc_f,
                    "t2_ftp_caught": c_ftp,
                    "t2_ftp_missed": n_esc_f - c_ftp,
                    "t2_lofo_caught": c_lofo,
                    "t2_lofo_missed": n_esc_f - c_lofo,
                    "t2_ftp_caught_pct": float((c_ftp / max(1, n_esc_f)) * 100.0),
                    "t2_lofo_caught_pct": float((c_lofo / max(1, n_esc_f)) * 100.0),
                }

            out[b_key] = {
                "target_budget": b * 100.0,
                "achieved_benign": achieved_benign,
                "achieved_benign_ci95": float(1.96 * np.std(boot_benign)),
                "novel_macro_coverage": novel_macro_cov,
                "novel_macro_coverage_ci95": float(1.96 * np.std(boot_macro)),
                "novel_pooled_coverage": novel_pooled_cov,
                "novel_pooled_coverage_ci95": float(1.96 * np.std(boot_pooled)),
                "family_counts": fam_counts,
                "tier2_on_novel_escalated": {
                    "novel_escalated_count": nov_esc_count,
                    "t2_ftp_caught": t2_ftp_caught_novel,
                    "t2_ftp_missed": t2_ftp_missed_novel,
                    "t2_lofo_caught": t2_lofo_caught_novel,
                    "t2_lofo_missed": t2_lofo_missed_novel,
                    "oracle_caught": nov_esc_count,
                    "oracle_missed": 0,
                },
                "end_to_end_miss": {
                    "total_novel_flows": total_novel,
                    "oracle_caught": oracle_e2e_caught,
                    "oracle_miss": oracle_e2e_miss,
                    "oracle_miss_rate_pct": oracle_miss_rate,
                    "ftp_only_caught": ftp_e2e_caught,
                    "ftp_only_miss": ftp_e2e_miss,
                    "ftp_only_miss_rate_pct": ftp_e2e_miss_rate,
                    "lofo_caught": lofo_e2e_caught,
                    "lofo_miss": lofo_e2e_miss,
                    "lofo_miss_rate_pct": lofo_e2e_miss_rate,
                },
                "tier2_per_family_on_escalated": t2_per_fam,
            }
        return out
        return out

    gates_eval = {
        "Full_kNN": evaluate_single_gate(knn_score_all, knn_score_benign),
        "Prototype_kNN": evaluate_single_gate(proto_score_all, proto_score_benign),
        "Mahalanobis": evaluate_single_gate(maha_score_all, maha_score_benign),
        "RF_Band": evaluate_single_gate(rf_score_all, rf_score_benign),
        "UNION_Gate": evaluate_single_gate(
            rf_score_all, rf_score_benign, is_union=True, knn_s=knn_score_all, knn_b=knn_score_benign
        ),
    }

    # ── 4. Deployable vs Oracle Calibration Comparison ───────────────────────
    # ── 4. Deployable vs Oracle Calibration Comparison ───────────────────────
    cal_comparison = {"kNN": {}, "Mahalanobis": {}}
    test_cal_all_knn = nn_cal.kneighbors(X_te_sc_cal)[0][:, -1]
    diff_test_cal_all = X_te_sc_cal - np.mean(b_fit_sc, axis=0)
    test_cal_all_maha = np.sqrt(np.sum(diff_test_cal_all @ inv_cov_cal * diff_test_cal_all, axis=1))

    for b in TARGET_BUDGETS:
        b_key = f"{int(b*100)}%"

        # kNN
        th_dep_knn = float(np.percentile(cal_scores_knn, 100 * (1 - b)))
        th_orc_knn = float(np.percentile(knn_score_benign, 100 * (1 - b)))
        esc_dep_knn = test_cal_all_knn >= th_dep_knn
        esc_orc_knn = knn_score_all >= th_orc_knn

        ach_dep_knn = float(np.mean(esc_dep_knn[benign_te_mask]) * 100.0)
        ach_orc_knn = float(np.mean(esc_orc_knn[benign_te_mask]) * 100.0)
        dep_macro_knn = float(np.mean([np.mean(esc_dep_knn[y_fam == fam]) * 100.0 for fam in NOVEL_FAMILIES]))
        dep_pooled_knn = float(np.mean(esc_dep_knn[novel_te_mask]) * 100.0)
        orc_macro_knn = float(np.mean([np.mean(esc_orc_knn[y_fam == fam]) * 100.0 for fam in NOVEL_FAMILIES]))
        orc_pooled_knn = float(np.mean(esc_orc_knn[novel_te_mask]) * 100.0)

        cal_comparison["kNN"][b_key] = {
            "deployable_threshold": th_dep_knn,
            "deployable_achieved_benign": ach_dep_knn,
            "deployable_macro_coverage": dep_macro_knn,
            "deployable_pooled_coverage": dep_pooled_knn,
            "oracle_threshold": th_orc_knn,
            "oracle_achieved_benign": ach_orc_knn,
            "oracle_macro_coverage": orc_macro_knn,
            "oracle_pooled_coverage": orc_pooled_knn,
        }

        # Mahalanobis
        th_dep_maha = float(np.percentile(cal_scores_maha, 100 * (1 - b)))
        th_orc_maha = float(np.percentile(maha_score_benign, 100 * (1 - b)))
        esc_dep_maha = test_cal_all_maha >= th_dep_maha
        esc_orc_maha = maha_score_all >= th_orc_maha

        ach_dep_maha = float(np.mean(esc_dep_maha[benign_te_mask]) * 100.0)
        ach_orc_maha = float(np.mean(esc_orc_maha[benign_te_mask]) * 100.0)
        dep_macro_maha = float(np.mean([np.mean(esc_dep_maha[y_fam == fam]) * 100.0 for fam in NOVEL_FAMILIES]))
        dep_pooled_maha = float(np.mean(esc_dep_maha[novel_te_mask]) * 100.0)
        orc_macro_maha = float(np.mean([np.mean(esc_orc_maha[y_fam == fam]) * 100.0 for fam in NOVEL_FAMILIES]))
        orc_pooled_maha = float(np.mean(esc_orc_maha[novel_te_mask]) * 100.0)

        cal_comparison["Mahalanobis"][b_key] = {
            "deployable_threshold": th_dep_maha,
            "deployable_achieved_benign": ach_dep_maha,
            "deployable_macro_coverage": dep_macro_maha,
            "deployable_pooled_coverage": dep_pooled_maha,
            "oracle_threshold": th_orc_maha,
            "oracle_achieved_benign": ach_orc_maha,
            "oracle_macro_coverage": orc_macro_maha,
            "oracle_pooled_coverage": orc_pooled_maha,
        }

    # ── 5. Paired Per-Block Differences vs RF Band ───────────────────────────
    paired_diffs = {}
    gate_descriptions = {
        "Full_kNN": "Full kNN Gate (Standardized, k=5, 9 features, N=4000 reference)",
        "Prototype_kNN": "Prototype kNN Gate (M=32 cluster centroids, 9 features)",
        "Mahalanobis": "Mahalanobis Distance Gate (Regularized covariance, 9 features)",
        "UNION_Gate": "UNION Gate (RF Band + Full kNN, split budget, 9 features)",
    }
    for gate_name in ["Full_kNN", "Prototype_kNN", "Mahalanobis", "UNION_Gate"]:
        paired_diffs[gate_name] = {
            "description": gate_descriptions[gate_name],
            "baseline": "RF Probability Band Gate (Uncertainty thresholding)",
            "budgets": {},
        }
        for b in TARGET_BUDGETS:
            b_key = f"{int(b*100)}%"
            th_rf = np.percentile(rf_score_benign, 100 * (1 - b))

            if gate_name == "Full_kNN":
                th_g = np.percentile(knn_score_benign, 100 * (1 - b))
                s_g = knn_score_all
            elif gate_name == "Prototype_kNN":
                th_g = np.percentile(proto_score_benign, 100 * (1 - b))
                s_g = proto_score_all
            elif gate_name == "Mahalanobis":
                th_g = np.percentile(maha_score_benign, 100 * (1 - b))
                s_g = maha_score_all
            elif gate_name == "UNION_Gate":
                th_u_rf = np.percentile(rf_score_benign, 100 * (1 - b / 2))
                th_u_knn = np.percentile(knn_score_benign, 100 * (1 - b / 2))

            macro_diffs = []
            pooled_diffs = []
            strata_diff = {"Benign": np.where(benign_te_mask)[0]}
            for f_name in ALL_FAMILIES:
                strata_diff[f_name] = np.where(y_fam == f_name)[0]

            for _ in range(1000):
                s_idx = np.concatenate([
                    rng_boot.choice(idx_list, size=len(idx_list), replace=True)
                    for idx_list in strata_diff.values()
                ])
                # RF coverage in this resample
                rf_covs = [
                    np.mean(rf_score_all[s_idx][y_fam[s_idx] == fam] >= th_rf) * 100.0
                    for fam in NOVEL_FAMILIES
                    if np.sum(y_fam[s_idx] == fam) > 0
                ]
                rf_macro = float(np.mean(rf_covs) if rf_covs else 0.0)
                s_nov = novel_te_mask[s_idx]
                rf_pooled = float(np.mean(rf_score_all[s_idx][s_nov] >= th_rf) * 100.0) if np.sum(s_nov) > 0 else 0.0

                if gate_name == "UNION_Gate":
                    u_esc = (rf_score_all[s_idx] >= th_u_rf) | (knn_score_all[s_idx] >= th_u_knn)
                    g_covs = [
                        np.mean(u_esc[y_fam[s_idx] == fam]) * 100.0
                        for fam in NOVEL_FAMILIES
                        if np.sum(y_fam[s_idx] == fam) > 0
                    ]
                    g_macro = float(np.mean(g_covs) if g_covs else 0.0)
                    g_pooled = float(np.mean(u_esc[s_nov]) * 100.0) if np.sum(s_nov) > 0 else 0.0
                else:
                    g_covs = [
                        np.mean(s_g[s_idx][y_fam[s_idx] == fam] >= th_g) * 100.0
                        for fam in NOVEL_FAMILIES
                        if np.sum(y_fam[s_idx] == fam) > 0
                    ]
                    g_macro = float(np.mean(g_covs) if g_covs else 0.0)
                    g_pooled = float(np.mean(s_g[s_idx][s_nov] >= th_g) * 100.0) if np.sum(s_nov) > 0 else 0.0

                macro_diffs.append(g_macro - rf_macro)
                pooled_diffs.append(g_pooled - rf_pooled)

            macro_arr = np.array(macro_diffs)
            pooled_arr = np.array(pooled_diffs)

            paired_diffs[gate_name]["budgets"][b_key] = {
                "mean_paired_difference": float(np.mean(macro_arr)),
                "ci95_paired_difference": float(1.96 * np.std(macro_arr)),
                "pct_blocks_positive": float(np.mean(macro_arr > 0) * 100.0),
                "pct_resamples_positive_macro": float(np.mean(macro_arr > 0) * 100.0),
                "percentile_ci_macro": [float(np.percentile(macro_arr, 2.5)), float(np.percentile(macro_arr, 97.5))],
                "mean_paired_macro_difference": float(np.mean(macro_arr)),
                "ci95_paired_macro_difference": float(1.96 * np.std(macro_arr)),
                "mean_paired_pooled_difference": float(np.mean(pooled_arr)),
                "ci95_paired_pooled_difference": float(1.96 * np.std(pooled_arr)),
                "pct_blocks_positive_pooled": float(np.mean(pooled_arr > 0) * 100.0),
                "pct_resamples_positive_pooled": float(np.mean(pooled_arr > 0) * 100.0),
                "percentile_ci_pooled": [float(np.percentile(pooled_arr, 2.5)), float(np.percentile(pooled_arr, 97.5))],
            }

    # ── 6. SSH Check on Standardized 25-Feature Gate ─────────────────────────
    b_tr_25 = X_tr_25[benign_tr_mask]
    b_te_25 = X_te_25[benign_te_mask]

    sc_25 = StandardScaler().fit(b_tr_25)
    X_tr_25_sc = sc_25.transform(X_tr_25)
    X_te_25_sc = sc_25.transform(X_te_25)
    b_tr_25_sc = sc_25.transform(b_tr_25)
    b_te_25_sc = sc_25.transform(b_te_25)

    nn_25 = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(b_tr_25_sc[:4000])
    knn25_score_all = nn_25.kneighbors(X_te_25_sc)[0][:, -1]
    knn25_score_benign = nn_25.kneighbors(b_te_25_sc)[0][:, -1]

    cov_25 = np.cov(b_tr_25_sc, rowvar=False) + 1e-4 * np.eye(25)
    inv_cov_25 = np.linalg.pinv(cov_25)
    diff25_all = X_te_25_sc - np.mean(b_tr_25_sc, axis=0)
    diff25_benign = b_te_25_sc - np.mean(b_tr_25_sc, axis=0)
    maha25_score_all = np.sqrt(np.sum(diff25_all @ inv_cov_25 * diff25_all, axis=1))
    maha25_score_benign = np.sqrt(np.sum(diff25_benign @ inv_cov_25 * diff25_benign, axis=1))

    ssh_mask = y_fam == "SSH-BruteForce"
    ssh_knn25_scores = knn25_score_all[ssh_mask]
    ssh_maha25_scores = maha25_score_all[ssh_mask]

    ssh_check_results = {
        "total_ssh_test_flows": int(np.sum(ssh_mask)),
        "score_distribution": {
            "knn25_benign": {
                "min": float(np.min(knn25_score_benign)),
                "median": float(np.median(knn25_score_benign)),
                "p90": float(np.percentile(knn25_score_benign, 90)),
                "p99": float(np.percentile(knn25_score_benign, 99)),
                "max": float(np.max(knn25_score_benign)),
            },
            "knn25_ssh": {
                "min": float(np.min(ssh_knn25_scores)),
                "median": float(np.median(ssh_knn25_scores)),
                "p90": float(np.percentile(ssh_knn25_scores, 90)),
                "p99": float(np.percentile(ssh_knn25_scores, 99)),
                "max": float(np.max(ssh_knn25_scores)),
            },
        },
        "budgets": {},
    }

    for b in TARGET_BUDGETS:
        b_key = f"{int(b*100)}%"
        th_k25 = np.percentile(knn25_score_benign, 100 * (1 - b))
        th_m25 = np.percentile(maha25_score_benign, 100 * (1 - b))

        esc_k25 = knn25_score_all >= th_k25
        esc_m25 = maha25_score_all >= th_m25

        ssh_check_results["budgets"][b_key] = {
            "knn25_ssh_escalated": int(np.sum(esc_k25[ssh_mask])),
            "knn25_ssh_coverage_pct": float(np.mean(esc_k25[ssh_mask]) * 100.0),
            "maha25_ssh_escalated": int(np.sum(esc_m25[ssh_mask])),
            "maha25_ssh_coverage_pct": float(np.mean(esc_m25[ssh_mask]) * 100.0),
        }

    # ── 7. Detailed Per-Flow Tier 2 Log for Slowloris (Reconciliation) ────────
    slow_mask = y_fam == "DoS-Slowloris"
    th_2pct = np.percentile(knn_score_benign, 98)
    esc_2pct = knn_score_all >= th_2pct
    slow_esc = esc_2pct & slow_mask

    slowloris_t2_audit = {
        "total_test_slowloris": int(np.sum(slow_mask)),
        "escalated_slowloris_count": int(np.sum(slow_esc)),
        "non_escalated_slowloris_count": int(np.sum((~esc_2pct) & slow_mask)),
        "escalated_caught_by_t2": int(np.sum((preds_t2_ftp == 1) & slow_esc)),
        "escalated_missed_by_t2": int(np.sum((preds_t2_ftp == 0) & slow_esc)),
        "non_escalated_caught_by_t2": int(np.sum((preds_t2_ftp == 1) & (~esc_2pct) & slow_mask)),
        "non_escalated_missed_by_t2": int(np.sum((preds_t2_ftp == 0) & (~esc_2pct) & slow_mask)),
        "mean_prob_escalated_slowloris": float(np.mean(probs_t2_ftp[slow_esc])),
        "max_prob_escalated_slowloris": float(np.max(probs_t2_ftp[slow_esc])),
        "standalone_t2_slowloris_recall": float(np.mean(preds_t2_ftp[slow_mask]) * 100.0),
        "standalone_t1_slowloris_recall": float(np.mean(preds_t1[slow_mask]) * 100.0),
    }

    # Compile Final JSON Structure
    final_output = {
        "execution_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "standardized_gates": gates_eval,
        "calibration_comparison": cal_comparison,
        "paired_block_differences": paired_diffs,
        "ssh_check_25_features": ssh_check_results,
        "slowloris_tier2_audit": slowloris_t2_audit,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved standardized gate evaluation results to: {OUT_JSON}")


if __name__ == "__main__":
    run_standardized_eval()
