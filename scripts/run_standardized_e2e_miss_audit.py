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

    # Train Tier 2 (RF on 76 features)
    rf_t2 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_tr_76, y_tr)
    preds_t2 = rf_t2.predict(X_te_76)
    probs_t2 = rf_t2.predict_proba(X_te_76)[:, 1]

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

            novel_cov = float(np.mean([fam_counts[fam]["coverage_pct"] for fam in NOVEL_FAMILIES]))

            # Block bootstrap over 10 time blocks
            boot_benign, boot_cov = [], []
            for _ in range(500):
                sampled_b = rng_boot.choice(N_BLOCKS, size=N_BLOCKS, replace=True)
                idx = np.concatenate([blocks[b_i] for b_i in sampled_b])
                b_esc = esc_mask[idx]
                b_y = y_te[idx]
                b_fam = y_fam[idx]
                boot_benign.append(float(np.mean(b_esc[b_y == 0]) * 100.0))
                covs = [
                    np.mean(b_esc[b_fam == fam]) * 100.0
                    for fam in NOVEL_FAMILIES
                    if np.sum(b_fam == fam) > 0
                ]
                boot_cov.append(float(np.mean(covs) if covs else 0.0))

            # Tier 2 Predictions on Escalated Novel Flows
            nov_esc = esc_mask & novel_te_mask
            nov_esc_count = int(np.sum(nov_esc))
            t2_caught_novel = int(np.sum((preds_t2 == 1) & nov_esc))
            t2_missed_novel = int(nov_esc_count - t2_caught_novel)

            # End-to-End Miss Rates on Novel Flows (N=1532)
            total_novel = int(np.sum(novel_te_mask))
            # Oracle Tier 2 catches all escalated novel flows
            oracle_e2e_caught = int(np.sum(((preds_t1 == 1) & (~esc_mask) & novel_te_mask) | nov_esc))
            oracle_e2e_miss = total_novel - oracle_e2e_caught
            oracle_miss_rate = float((oracle_e2e_miss / total_novel) * 100.0)

            # Trained Tier 2
            trained_e2e_caught = int(
                np.sum(((preds_t1 == 1) & (~esc_mask) & novel_te_mask) | ((preds_t2 == 1) & nov_esc))
            )
            trained_e2e_miss = total_novel - trained_e2e_caught
            trained_miss_rate = float((trained_e2e_miss / total_novel) * 100.0)

            # Detailed per-family Tier 2 audit on escalated flows
            t2_per_fam = {}
            for fam in ALL_FAMILIES:
                fm = y_fam == fam
                f_esc = esc_mask & fm
                n_esc_f = int(np.sum(f_esc))
                c_t2 = int(np.sum((preds_t2 == 1) & f_esc))
                m_t2 = n_esc_f - c_t2
                t2_per_fam[fam] = {
                    "escalated_count": n_esc_f,
                    "t2_caught": c_t2,
                    "t2_missed": m_t2,
                    "t2_caught_pct": float((c_t2 / max(1, n_esc_f)) * 100.0),
                }

            out[b_key] = {
                "target_budget": b * 100.0,
                "achieved_benign": achieved_benign,
                "achieved_benign_ci95": float(1.96 * np.std(boot_benign)),
                "novel_macro_coverage": novel_cov,
                "novel_macro_coverage_ci95": float(1.96 * np.std(boot_cov)),
                "family_counts": fam_counts,
                "tier2_on_novel_escalated": {
                    "novel_escalated_count": nov_esc_count,
                    "t2_caught": t2_caught_novel,
                    "t2_missed": t2_missed_novel,
                    "oracle_caught": nov_esc_count,
                    "oracle_missed": 0,
                },
                "end_to_end_miss": {
                    "total_novel_flows": total_novel,
                    "oracle_caught": oracle_e2e_caught,
                    "oracle_miss": oracle_e2e_miss,
                    "oracle_miss_rate_pct": oracle_miss_rate,
                    "trained_caught": trained_e2e_caught,
                    "trained_miss": trained_e2e_miss,
                    "trained_miss_rate_pct": trained_miss_rate,
                },
                "tier2_per_family_on_escalated": t2_per_fam,
            }
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
    cal_comparison = {"kNN": {}, "Mahalanobis": {}}
    for b in TARGET_BUDGETS:
        b_key = f"{int(b*100)}%"

        # kNN
        th_dep_knn = float(np.percentile(cal_scores_knn, 100 * (1 - b)))
        th_orc_knn = float(np.percentile(knn_score_benign, 100 * (1 - b)))
        ach_dep_knn = float(np.mean(test_cal_knn >= th_dep_knn) * 100.0)
        ach_orc_knn = float(np.mean(knn_score_benign >= th_orc_knn) * 100.0)

        cal_comparison["kNN"][b_key] = {
            "deployable_threshold": th_dep_knn,
            "deployable_achieved_benign": ach_dep_knn,
            "oracle_threshold": th_orc_knn,
            "oracle_achieved_benign": ach_orc_knn,
        }

        # Mahalanobis
        th_dep_maha = float(np.percentile(cal_scores_maha, 100 * (1 - b)))
        th_orc_maha = float(np.percentile(maha_score_benign, 100 * (1 - b)))
        ach_dep_maha = float(np.mean(test_cal_maha >= th_dep_maha) * 100.0)
        ach_orc_maha = float(np.mean(maha_score_benign >= th_orc_maha) * 100.0)

        cal_comparison["Mahalanobis"][b_key] = {
            "deployable_threshold": th_dep_maha,
            "deployable_achieved_benign": ach_dep_maha,
            "oracle_threshold": th_orc_maha,
            "oracle_achieved_benign": ach_orc_maha,
        }

    # ── 5. Paired Per-Block Differences vs RF Band ───────────────────────────
    paired_diffs = {}
    for gate_name in ["Full_kNN", "Prototype_kNN", "Mahalanobis", "UNION_Gate"]:
        paired_diffs[gate_name] = {}
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

            diffs = []
            for b_idx in blocks:
                # RF coverage in this block
                rf_covs = [
                    np.mean(rf_score_all[b_idx][y_fam[b_idx] == fam] >= th_rf) * 100.0
                    for fam in NOVEL_FAMILIES
                    if np.sum(y_fam[b_idx] == fam) > 0
                ]
                rf_c = float(np.mean(rf_covs) if rf_covs else 0.0)

                if gate_name == "UNION_Gate":
                    u_esc = (rf_score_all[b_idx] >= th_u_rf) | (knn_score_all[b_idx] >= th_u_knn)
                    g_covs = [
                        np.mean(u_esc[y_fam[b_idx] == fam]) * 100.0
                        for fam in NOVEL_FAMILIES
                        if np.sum(y_fam[b_idx] == fam) > 0
                    ]
                else:
                    g_covs = [
                        np.mean(s_g[b_idx][y_fam[b_idx] == fam] >= th_g) * 100.0
                        for fam in NOVEL_FAMILIES
                        if np.sum(y_fam[b_idx] == fam) > 0
                    ]
                g_c = float(np.mean(g_covs) if g_covs else 0.0)
                diffs.append(g_c - rf_c)

            diffs_arr = np.array(diffs)
            mean_d = float(np.mean(diffs_arr))
            ci_d = float(1.96 * np.std(diffs_arr) / np.sqrt(len(diffs_arr)))
            pos_pct = float(np.mean(diffs_arr > 0) * 100.0)

            paired_diffs[gate_name][b_key] = {
                "mean_paired_difference": mean_d,
                "ci95_paired_difference": ci_d,
                "pct_blocks_positive": pos_pct,
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
        "escalated_caught_by_t2": int(np.sum((preds_t2 == 1) & slow_esc)),
        "escalated_missed_by_t2": int(np.sum((preds_t2 == 0) & slow_esc)),
        "non_escalated_caught_by_t2": int(np.sum((preds_t2 == 1) & (~esc_2pct) & slow_mask)),
        "non_escalated_missed_by_t2": int(np.sum((preds_t2 == 0) & (~esc_2pct) & slow_mask)),
        "mean_prob_escalated_slowloris": float(np.mean(probs_t2[slow_esc])),
        "max_prob_escalated_slowloris": float(np.max(probs_t2[slow_esc])),
        "standalone_t2_slowloris_recall": float(np.mean(preds_t2[slow_mask]) * 100.0),
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
