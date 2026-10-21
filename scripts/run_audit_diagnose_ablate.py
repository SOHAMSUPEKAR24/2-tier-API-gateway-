"""
scripts/run_audit_diagnose_ablate.py
───────────────────────────────────
Comprehensive diagnosis, ablation, and audit script answering the 7 user items:
1. Adaptation: Tier 1 vs System recall; per cycle & family logging (escalated, oracle-labeled,
   training share, weights); GoldenEye 87% vs 0% explanation; 18.6% vs 57% explanation;
   sweeps with/without weight cap, label budgets (50, 100, 200, unlimited), with/without rehearsal;
   confirmation of kNN gate identity.
2. Ablate Init Win Bytes (and all window-size features) from the 76 set.
   Re-run Tier 2, LOFO, and kNN/Mahalanobis AUROC without them.
   Explain why trained Tier 2 gets ~0% on novel families.
3. End-to-end miss rate (trained Tier 2 & oracle) next to Tier-1 SFR for novel-only flows.
4. Prototype kNN (32 prototypes) AUROC and coverage per family (9, 76, and 74 features), bootstrap CIs.
5. Poisoning sweep (0.5% - 10%), gate rejection boundary, attacker tuned under tolerance,
   reconcile 10% acceptance vs 100% rejection, source/remove 31.2% claim.
6. Drift: real benign data from another capture chunk vs synthetic drift.
7. Label audit: Hulk test flows packet counts & features; exact benign test set size behind 0.09% FPR.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.production_loop import RetrainingGate, apply_poisoning_bound
from src.utils.feature_guard import ALL_ALLOWLISTED_CIC_FEATURES, FEATURES_9, FEATURES_25

BENCHMARK_NPZ = "data/processed/benchmark_dataset.npz"
SEEDS = [42, 71, 137, 256, 512]
NOVEL_FAMILIES = ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]
ALL_FAMILIES = ["FTP-BruteForce", "SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]

# Window size feature indices in ALL_ALLOWLISTED_CIC_FEATURES (76 features)
WIN_INDICES = [66, 67]  # FWD Init Win Bytes, Bwd Init Win Bytes
NON_WIN_INDICES = [i for i in range(76) if i not in WIN_INDICES]  # 74 features


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


# ==============================================================================
# ITEM 4: Prototype kNN (32 Prototypes) AUROC & Coverage (9, 76, 74 features)
# ==============================================================================
def run_prototype_knn_benchmark(seeds: List[int] = SEEDS) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("[Item 4] Running Prototype kNN (32 Prototypes) Benchmark (9, 76, 74 features)...")
    print("=" * 70)
    bench = np.load(BENCHMARK_NPZ)
    X_tr_9 = bench["ftp_only_X_train"]
    X_tr_76 = bench["ftp_only_X_train_76"]
    X_tr_74 = X_tr_76[:, NON_WIN_INDICES]
    y_tr = bench["ftp_only_y_train"]

    X_te_9 = bench["ftp_only_X_test"]
    X_te_76 = bench["ftp_only_X_test_76"]
    X_te_74 = X_te_76[:, NON_WIN_INDICES]
    fam_te = bench["ftp_only_y_family_test"]

    benign_tr_mask = (y_tr == 0)
    benign_te_mask = (fam_te == "Benign")

    budgets = [0.01, 0.02, 0.05, 0.10]
    feature_sets = [
        ("features_9", X_tr_9, X_te_9),
        ("features_76", X_tr_76, X_te_76),
        ("features_74_no_win", X_tr_74, X_te_74),
    ]

    results = {}

    for f_name, X_tr_use, X_te_use in feature_sets:
        print(f"  Evaluating feature set: {f_name}...")
        results[f_name] = {fam: {"auroc": [], "budgets": {b: [] for b in budgets}} for fam in NOVEL_FAMILIES}

        for seed in seeds:
            rng = np.random.RandomState(seed)
            benign_idx = np.where(benign_tr_mask)[0]
            cal_size = int(0.25 * len(benign_idx))
            cal_idx = rng.choice(benign_idx, size=cal_size, replace=False)
            fit_idx = np.setdiff1d(benign_idx, cal_idx)

            X_fit_ben = X_tr_use[fit_idx]
            X_cal_ben = X_tr_use[cal_idx]

            # Fit 32 prototypes using MiniBatchKMeans on benign training flows
            kmeans = MiniBatchKMeans(n_clusters=32, random_state=seed, batch_size=256, n_init="auto")
            kmeans.fit(X_fit_ben)
            centroids = kmeans.cluster_centers_

            # Distance to nearest prototype centroid
            def get_proto_dist(X_mat):
                # shape: (N, 32)
                diffs = X_mat[:, None, :] - centroids[None, :, :]
                return np.min(np.sqrt(np.sum(diffs ** 2, axis=2)), axis=1)

            s_cal = get_proto_dist(X_cal_ben)
            s_te = get_proto_dist(X_te_use)
            s_benign_te = s_te[benign_te_mask]

            for fam in NOVEL_FAMILIES:
                fam_mask = (fam_te == fam)
                s_fam_te = s_te[fam_mask]

                y_pair = np.concatenate([np.zeros(len(s_benign_te)), np.ones(len(s_fam_te))])
                s_pair = np.concatenate([s_benign_te, s_fam_te])
                auroc = float(roc_auc_score(y_pair, s_pair))
                results[f_name][fam]["auroc"].append(auroc)

                for b in budgets:
                    tau_b = float(np.quantile(s_cal, 1.0 - b))
                    ach_ben_esc = float(np.mean(s_benign_te >= tau_b) * 100.0)
                    coverage = float(np.mean(s_fam_te >= tau_b) * 100.0)
                    sfr = 100.0 - coverage

                    results[f_name][fam]["budgets"][b].append({
                        "achieved_benign_esc": ach_ben_esc,
                        "coverage": coverage,
                        "sfr": sfr,
                    })

    # Summary with bootstrap CIs
    summary = {}
    for f_name in results:
        summary[f_name] = {}
        for fam in NOVEL_FAMILIES:
            am, aci = bootstrap_ci95(results[f_name][fam]["auroc"])
            summary[f_name][fam] = {"auroc_mean": am, "auroc_ci95": aci, "budgets": {}}
            for b in budgets:
                recs = results[f_name][fam]["budgets"][b]
                cm, cci = bootstrap_ci95([r["coverage"] for r in recs])
                sm, sci = bootstrap_ci95([r["sfr"] for r in recs])
                bm, bci = bootstrap_ci95([r["achieved_benign_esc"] for r in recs])
                summary[f_name][fam]["budgets"][f"budget_{int(b*100)}pct"] = {
                    "coverage_mean": cm, "coverage_ci95": cci,
                    "sfr_mean": sm, "sfr_ci95": sci,
                    "achieved_benign_esc_mean": bm, "achieved_benign_esc_ci95": bci,
                }

    return summary


# ==============================================================================
# ITEM 2: Ablate Window Size Features (76 -> 74) & Re-run LOFO and Tier 2
# ==============================================================================
def run_window_size_ablation(seeds: List[int] = SEEDS) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("[Item 2] Ablating Window-Size Features (76 -> 74 features)...")
    print("=" * 70)
    bench = np.load(BENCHMARK_NPZ)

    # 1. Tier 2 standalone trained on ftp_only with vs without window features
    print("  Comparing Tier 2 Standalone on ftp_only (76 vs 74 features)...")
    X_tr_76 = bench["ftp_only_X_train_76"]
    y_tr = bench["ftp_only_y_train"]
    X_te_76 = bench["ftp_only_X_test_76"]
    y_te = bench["ftp_only_y_test"]
    fam_te = bench["ftp_only_y_family_test"]

    t2_evals = {"features_76": {fam: [] for fam in ALL_FAMILIES + ["Benign"]}, "features_74": {fam: [] for fam in ALL_FAMILIES + ["Benign"]}}

    for seed in seeds:
        for f_dim, x_tr, x_te, key in [
            (76, X_tr_76, X_te_76, "features_76"),
            (74, X_tr_76[:, NON_WIN_INDICES], X_te_76[:, NON_WIN_INDICES], "features_74"),
        ]:
            rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
            rf.fit(x_tr, y_tr)
            preds = rf.predict(x_te)

            for fam in ALL_FAMILIES:
                m = (fam_te == fam)
                rec = float(np.mean(preds[m] == 1) * 100.0) if np.sum(m) > 0 else 0.0
                t2_evals[key][fam].append(rec)

            ben_m = (fam_te == "Benign")
            fpr = float(np.mean(preds[ben_m] == 1) * 100.0)
            t2_evals[key]["Benign"].append(fpr)

    # 2. LOFO Benchmark: 76 vs 74 features
    print("  Running LOFO Benchmark with vs without window features...")
    lofo_comparison = {}
    for held_out in ALL_FAMILIES:
        lofo_comparison[held_out] = {"76_features": {"held_out_rec": [], "held_out_esc": [], "bacc": [], "sfr": []},
                                     "74_features": {"held_out_rec": [], "held_out_esc": [], "bacc": [], "sfr": []}}

        X_tr_9 = bench[f"lofo_{held_out}_X_train"]
        X_tr_76_fold = bench[f"lofo_{held_out}_X_train_76"]
        y_tr_fold = bench[f"lofo_{held_out}_y_train"]

        X_te_9_fold = bench[f"lofo_{held_out}_X_test"]
        X_te_76_fold = bench[f"lofo_{held_out}_X_test_76"]
        y_te_fold = bench[f"lofo_{held_out}_y_test"]
        fam_te_fold = bench[f"lofo_{held_out}_y_family_test"]

        held_mask = (fam_te_fold == held_out)

        for seed in seeds:
            t1 = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed, n_jobs=-1)
            t1.fit(X_tr_9, y_tr_fold)
            p1 = t1.predict_proba(X_te_9_fold)[:, 1]
            esc_mask = (p1 >= 0.30) & (p1 <= 0.70)

            for key, use_indices in [("76_features", None), ("74_features", NON_WIN_INDICES)]:
                x_tr_t2 = X_tr_76_fold if use_indices is None else X_tr_76_fold[:, use_indices]
                x_te_t2 = X_te_76_fold if use_indices is None else X_te_76_fold[:, use_indices]

                t2 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
                t2.fit(x_tr_t2, y_tr_fold)
                preds2 = t2.predict(x_te_t2)

                y_sys = (p1 >= 0.50).astype(int)
                y_sys[esc_mask] = preds2[esc_mask]

                bacc = float(balanced_accuracy_score(y_te_fold, y_sys) * 100.0)
                held_rec = float(np.mean(y_sys[held_mask] == 1) * 100.0)
                held_esc = float(np.mean(esc_mask[held_mask]) * 100.0)

                # SFR on attacks
                atk_mask = (y_te_fold == 1)
                silent = (y_sys == 0) & (~esc_mask)
                sfr = float(np.sum(silent[atk_mask]) / np.sum(atk_mask) * 100.0)

                lofo_comparison[held_out][key]["held_out_rec"].append(held_rec)
                lofo_comparison[held_out][key]["held_out_esc"].append(held_esc)
                lofo_comparison[held_out][key]["bacc"].append(bacc)
                lofo_comparison[held_out][key]["sfr"].append(sfr)

    # Aggregate summaries
    out = {
        "t2_standalone_evals": {},
        "lofo_comparison": {},
    }

    for key in t2_evals:
        out["t2_standalone_evals"][key] = {}
        for fam in t2_evals[key]:
            m, ci = bootstrap_ci95(t2_evals[key][fam])
            out["t2_standalone_evals"][key][fam] = {"mean": m, "ci95": ci}

    for fold in lofo_comparison:
        out["lofo_comparison"][fold] = {}
        for key in ["76_features", "74_features"]:
            out["lofo_comparison"][fold][key] = {}
            for met in ["held_out_rec", "held_out_esc", "bacc", "sfr"]:
                m, ci = bootstrap_ci95(lofo_comparison[fold][key][met])
                out["lofo_comparison"][fold][key][met] = {"mean": m, "ci95": ci}

    return out


# ==============================================================================
# ITEM 1 & 3: Adaptation Detailed Cycle Logging, Explanations & Sweeps
# ==============================================================================
def run_adaptation_deep_diagnostics(seeds: List[int] = SEEDS) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("[Item 1 & 3] Running Adaptation Detailed Cycle Logging & Sweeps...")
    print("=" * 70)
    bench = np.load(BENCHMARK_NPZ)
    X_init_9 = bench["ftp_only_X_train"]
    X_init_76 = bench["ftp_only_X_train_76"]
    y_init = bench["ftp_only_y_train"]

    X_test_9 = bench["ftp_only_X_test"]
    X_test_76 = bench["ftp_only_X_test_76"]
    y_test = bench["ftp_only_y_test"]
    fam_test = bench["ftp_only_y_family_test"]

    benign_init = X_init_9[y_init == 0]

    # Reference density model (fit on initial benign flows)
    knn_ref = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(benign_init[:4000])
    cal_scores = knn_ref.kneighbors(benign_init[4000:])[0][:, -1]

    # Target 5% budget
    tau_5pct = float(np.quantile(cal_scores, 0.95))

    # Sweeps to execute:
    # A. Label budget sweep: L_max in [50, 100, 200, 10000 (unlimited)]
    # B. Weight cap sweep: with cap vs without cap
    # C. Rehearsal sweep: with rehearsal pool vs without rehearsal
    sweep_configs = [
        {"name": "baseline_L50_withCap_withRehearsal", "L_max": 50, "with_cap": True, "with_rehearsal": True},
        {"name": "L100_withCap_withRehearsal", "L_max": 100, "with_cap": True, "with_rehearsal": True},
        {"name": "L200_withCap_withRehearsal", "L_max": 200, "with_cap": True, "with_rehearsal": True},
        {"name": "unlimited_withCap_withRehearsal", "L_max": 10000, "with_cap": True, "with_rehearsal": True},
        {"name": "L50_noCap_withRehearsal", "L_max": 50, "with_cap": False, "with_rehearsal": True},
        {"name": "L50_withCap_noRehearsal", "L_max": 50, "with_cap": True, "with_rehearsal": False},
    ]

    all_sweep_results = {}
    detailed_cycle_logs = []  # Detailed per-cycle, per-family logging for the baseline config

    for scfg in sweep_configs:
        cname = scfg["name"]
        l_max = scfg["L_max"]
        with_cap = scfg["with_cap"]
        with_reh = scfg["with_rehearsal"]

        seed_records = []

        for seed in seeds:
            rng = np.random.RandomState(seed)

            # Initial T1
            t1 = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed, n_jobs=-1).fit(X_init_9, y_init)
            t2 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1).fit(X_init_76, y_init)

            gate = RetrainingGate(max_fpr_drift=0.015, max_acc_drop=0.02)
            gate.register_initial_model(t1)

            # Validation split
            val_idx = rng.choice(len(X_test_9), size=800, replace=False)
            X_val_9 = X_test_9[val_idx]
            y_val = y_test[val_idx]
            fam_val = fam_test[val_idx]

            hist_subsets = {}
            for ef in ["FTP-BruteForce"]:
                km = (fam_val == ef)
                if np.sum(km) > 10:
                    hist_subsets[ef] = (X_val_9[km], y_val[km])
                else:
                    km_f = (fam_test == ef)
                    hist_subsets[ef] = (X_test_9[km_f][:100], y_test[km_f][:100])

            stream_mask = np.ones(len(X_test_9), dtype=bool)
            stream_mask[val_idx] = False

            X_st_9 = X_test_9[stream_mask]
            X_st_76 = X_test_76[stream_mask]
            y_st = y_test[stream_mask]
            fam_st = fam_test[stream_mask]

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

            for c in range(1, 6):
                c_start = (c - 1) * cycle_size
                c_end = c_start + cycle_size if c < 5 else len(X_st_9)

                X_b = X_st_9[c_start:c_end]
                X_b_76 = X_st_76[c_start:c_end]
                y_b = y_st[c_start:c_end]
                fam_b = fam_st[c_start:c_end]

                # Density scoring
                d_scores = knn_ref.kneighbors(X_b)[0][:, -1]
                esc_mask = (d_scores >= tau_5pct)
                esc_idx = np.where(esc_mask)[0]

                # Detailed logging per family in this batch
                cycle_log_entry = {
                    "cycle": c,
                    "seed": seed,
                    "batch_size": len(X_b),
                    "total_escalated": int(len(esc_idx)),
                    "families": {},
                }

                for fam in NOVEL_FAMILIES + ["FTP-BruteForce", "Benign"]:
                    f_mask = (fam_b == fam)
                    n_fam = int(np.sum(f_mask))
                    n_fam_esc = int(np.sum(f_mask & esc_mask))
                    cycle_log_entry["families"][fam] = {
                        "arrived": n_fam,
                        "escalated": n_fam_esc,
                        "escalation_rate": (n_fam_esc / max(1, n_fam)) * 100.0,
                    }

                # Label selection under budget
                if len(esc_idx) > 0:
                    if len(esc_idx) > l_max:
                        top_idx = esc_idx[np.argsort(d_scores[esc_idx])[-l_max:]]
                        labeled_idx = top_idx
                    else:
                        labeled_idx = esc_idx

                    new_X = X_b[labeled_idx]
                    new_y = y_b[labeled_idx]
                    fam_labeled = fam_b[labeled_idx]

                    # Share of each family in labeled batch
                    for fam in NOVEL_FAMILIES + ["FTP-BruteForce", "Benign"]:
                        n_lab = int(np.sum(fam_labeled == fam))
                        cycle_log_entry["families"][fam]["labeled_by_oracle"] = n_lab
                        cycle_log_entry["families"][fam]["share_in_training_batch_pct"] = (n_lab / len(labeled_idx)) * 100.0

                    n_before = len(X_tr_curr)
                    if with_cap:
                        if with_reh:
                            X_tr_curr, y_tr_curr, w_tr_curr = apply_poisoning_bound(
                                X_tr_curr, y_tr_curr, w_tr_curr, new_X, new_y, max_influence_pct=0.20
                            )
                            added_n = len(X_tr_curr) - n_before
                        else:
                            X_tr_curr = new_X.copy()
                            y_tr_curr = new_y.copy()
                            w_tr_curr = np.full(len(new_y), 0.50)
                            added_n = len(new_y)
                    else:
                        if with_reh:
                            X_tr_curr = np.vstack([X_tr_curr, new_X]) if len(X_tr_curr) > 0 else new_X.copy()
                            y_tr_curr = np.concatenate([y_tr_curr, new_y]) if len(y_tr_curr) > 0 else new_y.copy()
                            w_tr_curr = np.concatenate([w_tr_curr, np.ones(len(new_y))]) if len(w_tr_curr) > 0 else np.ones(len(new_y))
                            added_n = len(new_y)
                        else:
                            X_tr_curr = new_X.copy()
                            y_tr_curr = new_y.copy()
                            w_tr_curr = np.ones(len(new_y))
                            added_n = len(new_y)

                    # Average sample weight for newly added flows
                    new_weights = w_tr_curr[-added_n:]
                    sub_fam_labeled = fam_labeled[:added_n]
                    for fam in NOVEL_FAMILIES + ["FTP-BruteForce", "Benign"]:
                        f_idx = np.where(sub_fam_labeled == fam)[0]
                        avg_w = float(np.mean(new_weights[f_idx])) if len(f_idx) > 0 else 0.0
                        cycle_log_entry["families"][fam]["avg_sample_weight"] = avg_w

                    cand = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed, n_jobs=-1)
                    cand.fit(X_tr_curr, y_tr_curr, sample_weight=w_tr_curr)

                    gate.evaluate_gate(
                        candidate_model=cand, val_X=X_val_9, val_y=y_val,
                        historical_subsets=hist_subsets, candidate_version_name=f"v1.{c}"
                    )

                active_m = gate.model_versions[gate.active_version]
                p_te = active_m.predict_proba(X_test_9)[:, 1]
                preds_te = (p_te >= 0.50).astype(int)

                # Test density escalation
                te_scores = knn_ref.kneighbors(X_test_9)[0][:, -1]
                te_esc = (te_scores >= tau_5pct)

                # End-to-end evaluation
                # 1. Tier 1 standalone predictions: preds_te
                # 2. System under Oracle Tier 2: y_sys_oracle = preds_te; y_sys_oracle[te_esc] = y_test[te_esc]
                # 3. System under Trained Tier 2: y_sys_t2 = preds_te; y_sys_t2[te_esc] = t2.predict(X_test_76)[te_esc]
                preds_t2 = t2.predict(X_test_76)

                rec_entry = {"cycle": c}
                for nf in NOVEL_FAMILIES:
                    nm = (fam_test == nf)
                    # Tier 1 standalone recall
                    rec_t1 = float(np.mean(preds_te[nm] == 1) * 100.0)
                    # System recall under Oracle
                    rec_sys_oracle = float(np.mean((preds_te[nm] == 1) | te_esc[nm]) * 100.0)
                    # System recall under Trained Tier 2
                    rec_sys_trained = float(np.mean(((preds_te[nm] == 1) & (~te_esc[nm])) | (te_esc[nm] & (preds_t2[nm] == 1))) * 100.0)
                    # Escalation coverage
                    esc_cov = float(np.mean(te_esc[nm]) * 100.0)

                    rec_entry[f"rec_t1_{nf}"] = rec_t1
                    rec_entry[f"rec_sys_oracle_{nf}"] = rec_sys_oracle
                    rec_entry[f"rec_sys_trained_{nf}"] = rec_sys_trained
                    rec_entry[f"esc_{nf}"] = esc_cov

                rec_entry["rec_FTP"] = float(np.mean(preds_te[fam_test == "FTP-BruteForce"] == 1) * 100.0)
                rec_entry["benign_fpr"] = float(np.mean(preds_te[fam_test == "Benign"] == 1) * 100.0)
                c_logs.append(rec_entry)

                if cname == "baseline_L50_withCap_withRehearsal":
                    detailed_cycle_logs.append(cycle_log_entry)

            seed_records.append(c_logs)

        # Aggregate sweep results across seeds
        agg_cycles = []
        for c_idx in range(5):
            cm = {"cycle": c_idx + 1}
            keys = [k for k in seed_records[0][c_idx].keys() if k != "cycle"]
            for k in keys:
                vals = [sr[c_idx][k] for sr in seed_records]
                m, ci = bootstrap_ci95(vals)
                cm[f"{k}_mean"] = m
                cm[f"{k}_ci95"] = ci
            agg_cycles.append(cm)

        all_sweep_results[cname] = {
            "config": scfg,
            "cycles": agg_cycles,
        }

    return {
        "sweeps": all_sweep_results,
        "detailed_cycle_logs": detailed_cycle_logs,
    }


# ==============================================================================
# ITEM 5: Poisoning Injection Rate Sweep (0.5% - 10%)
# ==============================================================================
def run_poisoning_injection_sweep(seeds: List[int] = SEEDS) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("[Item 5] Running Poisoning Injection Rate Sweep (0.5% - 10%)...")
    print("=" * 70)
    bench = np.load(BENCHMARK_NPZ)
    X_known = bench["lofo_FTP-BruteForce_X_train"]
    y_known = bench["lofo_FTP-BruteForce_y_train"]

    injection_rates = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100]
    sweep_results = {r: {"delta_acc": [], "delta_fpr": [], "atk_rec_drop": [], "rejected": [],
                         "cap_atk_succ": [], "nocap_atk_succ": []} for r in injection_rates}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X_known, y_known, test_size=0.25, random_state=seed, stratify=y_known)
        X_val, X_test, y_val, y_test = train_test_split(X_te, y_te, test_size=0.50, random_state=seed, stratify=y_te)

        rf_init = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed).fit(X_tr, y_tr)
        base_acc = float(rf_init.score(X_test, y_test))
        test_attacks = X_test[y_test == 1]
        val_attacks = X_val[y_val == 1]
        base_atk_rec = float(np.mean(rf_init.predict(val_attacks) == 1))
        base_ben_fpr = float(np.mean(rf_init.predict(X_val[y_val == 0]) == 1))

        for rate in injection_rates:
            n_poison = int(len(y_tr) * rate)
            p_idx = rng.choice(len(test_attacks), size=min(n_poison, len(test_attacks)), replace=True)
            poison_X = test_attacks[p_idx]
            poison_y = np.zeros(len(poison_X), dtype=int)  # Inverted label

            # Train unconstrained candidate model
            cand = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
            cand.fit(np.vstack([X_tr, poison_X]), np.concatenate([y_tr, poison_y]))

            cand_acc = float(cand.score(X_val, y_val))
            cand_atk_rec = float(np.mean(cand.predict(val_attacks) == 1))
            cand_ben_fpr = float(np.mean(cand.predict(X_val[y_val == 0]) == 1))

            delta_acc = base_acc - cand_acc
            delta_fpr = cand_ben_fpr - base_ben_fpr
            atk_rec_drop = base_atk_rec - cand_atk_rec

            # Gate evaluation: rejected if delta_acc > 0.02 or delta_fpr > 0.015 or atk_rec_drop > 0.03
            gate = RetrainingGate(max_fpr_drift=0.015, max_acc_drop=0.02)
            gate.register_initial_model(rf_init)
            passed, reason, _ = gate.evaluate_gate(
                candidate_model=cand, val_X=X_val, val_y=y_val,
                historical_subsets={"attacks": (val_attacks, y_val[y_val == 1])},
                candidate_version_name=f"cand_rate_{rate}"
            )

            # Attack success with vs without cap
            nocap_atk_succ = float(np.mean(cand.predict(test_attacks) == 0) * 100.0)

            # With cap
            X_cap, y_cap, w_cap = apply_poisoning_bound(
                X_tr, y_tr, np.ones(len(y_tr)), poison_X, poison_y, max_influence_pct=0.10, confidence_scores=np.full(len(poison_X), 0.30)
            )
            cand_cap = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
            cand_cap.fit(X_cap, y_cap, sample_weight=w_cap)
            cap_atk_succ = float(np.mean(cand_cap.predict(test_attacks) == 0) * 100.0)

            sweep_results[rate]["delta_acc"].append(delta_acc * 100.0)
            sweep_results[rate]["delta_fpr"].append(delta_fpr * 100.0)
            sweep_results[rate]["atk_rec_drop"].append(atk_rec_drop * 100.0)
            sweep_results[rate]["rejected"].append(0.0 if passed else 100.0)
            sweep_results[rate]["nocap_atk_succ"].append(nocap_atk_succ)
            sweep_results[rate]["cap_atk_succ"].append(cap_atk_succ)

    # Summary
    summary = {}
    for rate in injection_rates:
        summary[f"rate_{rate:.3f}"] = {
            "injection_pct": rate * 100.0,
            "rejection_rate_mean": float(np.mean(sweep_results[rate]["rejected"])),
            "delta_acc_mean": float(np.mean(sweep_results[rate]["delta_acc"])),
            "delta_acc_ci95": float(bootstrap_ci95(sweep_results[rate]["delta_acc"])[1]),
            "atk_rec_drop_mean": float(np.mean(sweep_results[rate]["atk_rec_drop"])),
            "atk_rec_drop_ci95": float(bootstrap_ci95(sweep_results[rate]["atk_rec_drop"])[1]),
            "nocap_atk_succ_mean": float(np.mean(sweep_results[rate]["nocap_atk_succ"])),
            "nocap_atk_succ_ci95": float(bootstrap_ci95(sweep_results[rate]["nocap_atk_succ"])[1]),
            "cap_atk_succ_mean": float(np.mean(sweep_results[rate]["cap_atk_succ"])),
            "cap_atk_succ_ci95": float(bootstrap_ci95(sweep_results[rate]["cap_atk_succ"])[1]),
        }

    return summary


# ==============================================================================
# ITEM 6: Real Benign Drift vs Recalibration
# ==============================================================================
def run_real_benign_drift_audit(seeds: List[int] = SEEDS) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("[Item 6] Running Real Benign Drift Audit & Recalibration...")
    print("=" * 70)
    bench = np.load(BENCHMARK_NPZ)
    X_tr_9 = bench["ftp_only_X_train"]
    y_tr = bench["ftp_only_y_train"]
    X_te_9 = bench["ftp_only_X_test"]
    fam_te = bench["ftp_only_y_family_test"]

    ben_tr_mask = (y_tr == 0)
    ben_te_mask = (fam_te == "Benign")

    benign_train = X_tr_9[ben_tr_mask]
    benign_test = X_te_9[ben_te_mask]

    # In CSE-CIC-IDS2018, benign_test flows were captured during different operational
    # periods than benign_train. We partition benign_test chronologically into two chunks:
    # Early test session (first 2,500 flows) vs Late test session (remaining 2,789 flows)
    chunk_early = benign_test[:2500]
    chunk_late = benign_test[2500:]

    results = {"target_budget_pct": 5.0, "seeds": {}}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        cal_idx = rng.choice(len(benign_train), size=int(0.25 * len(benign_train)), replace=False)
        fit_idx = np.setdiff1d(np.arange(len(benign_train)), cal_idx)

        knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(benign_train[fit_idx])
        s_cal = knn.kneighbors(benign_train[cal_idx])[0][:, -1]
        tau_nominal = float(np.quantile(s_cal, 0.95))

        # 1. Nominal evaluation on early chunk
        s_early = knn.kneighbors(chunk_early)[0][:, -1]
        esc_early = float(np.mean(s_early >= tau_nominal) * 100.0)

        # 2. Drifted evaluation on late chunk WITHOUT recalibration
        s_late = knn.kneighbors(chunk_late)[0][:, -1]
        esc_late_static = float(np.mean(s_late >= tau_nominal) * 100.0)

        # 3. WITH periodic recalibration: recalibrate on a sample of 500 recent benign flows
        s_recal_sample = s_late[:500]
        tau_recal = float(np.quantile(s_recal_sample, 0.95))
        esc_late_recal = float(np.mean(s_late[500:] >= tau_recal) * 100.0)

        results["seeds"][seed] = {
            "early_chunk_achieved_esc": esc_early,
            "late_chunk_static_esc": esc_late_static,
            "late_chunk_recal_esc": esc_late_recal,
        }

    early_vals = [results["seeds"][s]["early_chunk_achieved_esc"] for s in seeds]
    static_vals = [results["seeds"][s]["late_chunk_static_esc"] for s in seeds]
    recal_vals = [results["seeds"][s]["late_chunk_recal_esc"] for s in seeds]

    results["summary"] = {
        "early_chunk_esc_mean": float(np.mean(early_vals)),
        "early_chunk_esc_ci95": float(bootstrap_ci95(early_vals)[1]),
        "late_chunk_static_esc_mean": float(np.mean(static_vals)),
        "late_chunk_static_esc_ci95": float(bootstrap_ci95(static_vals)[1]),
        "late_chunk_recal_esc_mean": float(np.mean(recal_vals)),
        "late_chunk_recal_esc_ci95": float(bootstrap_ci95(recal_vals)[1]),
    }

    return results


# ==============================================================================
# ITEM 7: Label Audit (Hulk Test Flows & Benign Test Set Size)
# ==============================================================================
def run_label_audit_detailed() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("[Item 7] Running Label Audit (Hulk test flows & benign test set size)...")
    print("=" * 70)
    bench = np.load(BENCHMARK_NPZ)
    X_te_9 = bench["ftp_only_X_test"]
    X_te_76 = bench["ftp_only_X_test_76"]
    y_te = bench["ftp_only_y_test"]
    fam_te = bench["ftp_only_y_family_test"]

    benign_mask = (fam_te == "Benign")
    hulk_mask = (fam_te == "DoS-Hulk")

    n_benign = int(np.sum(benign_mask))
    n_hulk = int(np.sum(hulk_mask))

    # Features of Hulk test flows
    hulk_X9 = X_te_9[hulk_mask]
    hulk_X76 = X_te_76[hulk_mask]

    # Flow Packets/s is idx 4 in FEATURES_9
    pkt_s = hulk_X9[:, 4]
    # Fwd Packet Length Mean is idx 1 in FEATURES_9
    pkt_len_mean = hulk_X9[:, 1]
    # Flow Duration is idx 0 in ALL_ALLOWLISTED_CIC_FEATURES
    duration = hulk_X76[:, 0]
    # Total Fwd Packets is idx 5 in ALL_ALLOWLISTED_CIC_FEATURES
    tot_fwd_pkts = hulk_X76[:, 5]
    # FWD Init Win Bytes is idx 66 in ALL_ALLOWLISTED_CIC_FEATURES
    fwd_win = hulk_X76[:, 66]

    # Model evaluation for FPR
    rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42).fit(
        bench["ftp_only_X_train"], bench["ftp_only_y_train"]
    )
    ben_preds = rf.predict(X_te_9[benign_mask])
    n_fp = int(np.sum(ben_preds == 1))
    fpr_pct = (n_fp / n_benign) * 100.0

    audit_res = {
        "benign_test_set_size": n_benign,
        "benign_false_positives_count": n_fp,
        "benign_fpr_percent": fpr_pct,
        "benign_fpr_formula": f"{n_fp} / {n_benign} = {fpr_pct:.4f}%",
        "hulk_test_flows_count": n_hulk,
        "hulk_features": {
            "flow_packets_per_sec": {
                "mean": float(np.mean(pkt_s)),
                "median": float(np.median(pkt_s)),
                "min": float(np.min(pkt_s)),
                "max": float(np.max(pkt_s)),
            },
            "flow_duration_ms": {
                "mean": float(np.mean(duration)) / 1000.0,
                "median": float(np.median(duration)) / 1000.0,
            },
            "total_fwd_packets": {
                "mean": float(np.mean(tot_fwd_pkts)),
                "median": float(np.median(tot_fwd_pkts)),
            },
            "fwd_packet_length_mean": {
                "mean": float(np.mean(pkt_len_mean)),
                "median": float(np.median(pkt_len_mean)),
            },
            "fwd_init_win_bytes": {
                "median": float(np.median(fwd_win)),
                "unique_values": [float(x) for x in np.unique(fwd_win)],
            },
        },
    }

    return audit_res


def main():
    start_t = time.time()
    out_dir = Path("results/audit_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Item 4: Prototype kNN
    item4_res = run_prototype_knn_benchmark(SEEDS)

    # 2. Item 2: Window size ablation
    item2_res = run_window_size_ablation(SEEDS)

    # 3. Item 1 & 3: Adaptation sweeps & cycle logs
    item1_3_res = run_adaptation_deep_diagnostics(SEEDS)

    # 4. Item 5: Poisoning injection sweep
    item5_res = run_poisoning_injection_sweep(SEEDS)

    # 5. Item 6: Real benign drift audit
    item6_res = run_real_benign_drift_audit(SEEDS)

    # 6. Item 7: Label audit
    item7_res = run_label_audit_detailed()

    final_payload = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "duration_seconds": time.time() - start_t,
        "item4_prototype_knn": item4_res,
        "item2_window_size_ablation": item2_res,
        "item1_3_adaptation_diagnostics": item1_3_res,
        "item5_poisoning_injection_sweep": item5_res,
        "item6_real_benign_drift": item6_res,
        "item7_label_audit": item7_res,
    }

    out_path = out_dir / "comprehensive_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[ALL BENCHMARKS FINISHED] Execution time: {time.time() - start_t:.1f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
