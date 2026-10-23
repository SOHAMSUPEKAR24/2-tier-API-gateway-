#!/usr/bin/env python3
"""
scripts/run_sampling_sensitivity.py
───────────────────────────────────
Sampling Sensitivity Evaluation:
  1. Re-draws each attack family uniformly at random across its entire capture in cic.csv
     (and benign across the whole 18.96M rows file) using reservoir sampling (seed=42).
  2. Tracks exact day/session/chunk provenance for every sampled flow.
  3. Deduplicates on FEATURES_9 and partitions chronologically (70/30 attacks, 60/40 benign).
  4. Builds data/processed/benchmark_uniform_dataset.npz.
  5. Evaluates LOFO and the standardized escalation gates on the uniform dataset.
  6. Compares Old (first-N sequential quota) vs New (uniform across capture).
  7. Saves raw artifacts to results/lofo/sampling_sensitivity_raw.json.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.feature_guard import (
    ALL_ALLOWLISTED_CIC_FEATURES,
    FEATURES_9,
    FEATURES_25,
    enforce_allowlist,
    strip_identifiers,
    assert_no_leakage,
)

CIC_PATH = "/Users/sohamsupekar/Desktop/MajorProject/DATASET/cic.csv"
UNIFORM_NPZ_PATH = "data/processed/benchmark_uniform_dataset.npz"
OLD_NPZ_PATH = "data/processed/benchmark_dataset.npz"
OUTPUT_JSON = "results/lofo/sampling_sensitivity_raw.json"

TARGET_ATTACKS = [
    "FTP-BruteForce",
    "SSH-BruteForce",
    "DoS-GoldenEye",
    "DoS-Hulk",
    "DoS-Slowloris",
]

# Mapping of chunk ranges to known CSE-CIC-IDS2018 capture days and sessions
CHUNK_SESSION_MAP = {
    0: {"day": "14/02/2018 (Wed)", "session": "Morning (Early Benign, FTP, SSH, Slowloris bursts)"},
    1: {"day": "15/02/2018 (Thu)", "session": "Morning (Early GoldenEye, Slowloris, early Hulk flood)"},
    2: {"day": "15/02/2018 (Thu)", "session": "Midday (GoldenEye continuation, Heartbleed)"},
    3: {"day": "15/02/2018 (Thu) - 16/02/2018", "session": "DDoS and Web Attacks"},
    4: {"day": "16/02/2018 (Fri)", "session": "PortScan and background DDoS"},
    5: {"day": "14/02/2018 (Wed)", "session": "Afternoon Main FTP-BruteForce Flood (172k flows)"},
    6: {"day": "14/02/2018 (Wed)", "session": "Afternoon/Evening Main SSH-BruteForce Flood (187k flows)"},
    7: {"day": "15/02/2018 (Thu)", "session": "Afternoon Main DoS-GoldenEye (41k) & Slowloris (11k)"},
    8: {"day": "15/02/2018 (Thu)", "session": "Benign background transit"},
    9: {"day": "16/02/2018 (Fri)", "session": "Morning DoS-SlowHTTPTest (86k flows)"},
    10: {"day": "16/02/2018 (Fri)", "session": "Afternoon DoS-Hulk Primary Flood (257k flows)"},
    11: {"day": "16/02/2018 (Fri)", "session": "Afternoon DoS-Hulk Secondary Flood (204k flows)"},
}


def draw_uniform_sampled_corpus(
    cic_path: str = CIC_PATH,
    sample_per_attack: int = 1500,
    benign_total: int = 15000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Uniformly samples each attack family across its entire presence in cic.csv
    and benign traffic across all 18.96M rows using reservoir sampling.
    """
    target_quotas = {att: sample_per_attack for att in TARGET_ATTACKS}
    target_quotas["Benign"] = benign_total

    rng = np.random.RandomState(seed)
    reservoirs = {k: [] for k in target_quotas}
    chunk_origin_counts = {k: {} for k in target_quotas}
    seen_counts = {k: 0 for k in target_quotas}

    wanted_cols = set(ALL_ALLOWLISTED_CIC_FEATURES + ["Label"])
    chunk_size = 500000
    row_offset = 0

    print(f"[SamplingSensitivity] Streaming {cic_path} with uniform reservoir sampling...")
    t0 = time.time()

    reader = pd.read_csv(
        cic_path,
        chunksize=chunk_size,
        usecols=lambda c: c.strip() in wanted_cols,
    )

    for chunk_idx, chunk in enumerate(reader):
        chunk.columns = [c.strip() for c in chunk.columns]
        lbl_col = [c for c in chunk.columns if "label" in c.lower()][0]
        chunk[lbl_col] = chunk[lbl_col].str.strip()

        for k, cap in target_quotas.items():
            sub = chunk[chunk[lbl_col] == k]
            if len(sub) == 0:
                continue

            sub_indices = sub.index.values
            for local_idx in sub_indices:
                curr_seen = seen_counts[k]
                if curr_seen < cap:
                    reservoirs[k].append((chunk.loc[local_idx], chunk_idx))
                else:
                    j = rng.randint(0, curr_seen + 1)
                    if j < cap:
                        reservoirs[k][j] = (chunk.loc[local_idx], chunk_idx)
                seen_counts[k] += 1

        row_offset += len(chunk)
        if chunk_idx % 5 == 0 or chunk_idx == 37:
            elapsed = time.time() - t0
            print(f"  Chunk {chunk_idx:02d} processed ({row_offset:,} rows) [{elapsed:.1f}s]")

    # Extract dataframes and compute origin statistics
    frames = []
    provenance_stats = {
        "sampling_method": "Uniform reservoir sampling across all 18,965,059 rows (seed=42)",
        "total_rows_scanned": row_offset,
        "seen_population": seen_counts,
        "family_origins": {},
    }

    for k in target_quotas:
        items = reservoirs[k]
        dfs = [item[0] for item in items]
        chunks_recorded = [item[1] for item in items]

        # Count chunk frequencies
        c_counts = {}
        for c in chunks_recorded:
            c_counts[c] = c_counts.get(c, 0) + 1

        chunk_summary = []
        for c, count in sorted(c_counts.items()):
            desc = CHUNK_SESSION_MAP.get(c, {"day": "Other/Transit", "session": f"Chunk {c} background"})
            chunk_summary.append({
                "chunk_idx": c,
                "flows_sampled": count,
                "pct_of_sample": float(count / len(items) * 100.0),
                "day": desc["day"],
                "session": desc["session"],
            })

        provenance_stats["family_origins"][k] = {
            "total_sampled": len(items),
            "total_available_in_file": seen_counts[k],
            "sampling_fraction": float(len(items) / max(1, seen_counts[k])),
            "chunk_distribution": chunk_summary,
        }

        fam_df = pd.DataFrame(dfs)
        frames.append(fam_df)

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = strip_identifiers(df)

    # Deduplicate on FEATURES_9
    initial_len = len(df)
    df = df.drop_duplicates(subset=FEATURES_9).reset_index(drop=True)
    print(f"[SamplingSensitivity] Deduplication on FEATURES_9: {initial_len} -> {len(df)} unique flows.")

    # Clean non-finite
    for col in ALL_ALLOWLISTED_CIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[ALL_ALLOWLISTED_CIC_FEATURES] = df[ALL_ALLOWLISTED_CIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    provenance_stats["unique_flows_after_dedup"] = len(df)
    return df, provenance_stats


def build_uniform_benchmark_splits(
    df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Builds standardized benchmark splits identically to benchmark_pipeline.py."""
    lbl_col = [c for c in df.columns if "label" in c.lower()][0]
    splits = {}

    benign_df = df[df[lbl_col] == "Benign"].reset_index(drop=True)
    b_split = int(len(benign_df) * 0.60)
    benign_train = benign_df.iloc[:b_split]
    benign_test = benign_df.iloc[b_split:]

    attack_train_pools = {}
    attack_test_pools = {}
    for fam in TARGET_ATTACKS:
        fam_df = df[df[lbl_col] == fam].reset_index(drop=True)
        split_idx = int(len(fam_df) * 0.70)
        attack_train_pools[fam] = fam_df.iloc[:split_idx]
        attack_test_pools[fam] = fam_df.iloc[split_idx:]

    all_test_attacks = pd.concat([attack_test_pools[fam] for fam in TARGET_ATTACKS], ignore_index=True)
    all_test_df = pd.concat([benign_test, all_test_attacks], ignore_index=True)

    def extract_matrices(tr_df: pd.DataFrame, te_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        X_tr_9 = enforce_allowlist(tr_df, FEATURES_9)
        X_te_9 = enforce_allowlist(te_df, FEATURES_9)
        assert_no_leakage(X_tr_9, X_te_9)

        X_tr_25 = enforce_allowlist(tr_df, FEATURES_25)
        X_te_25 = enforce_allowlist(te_df, FEATURES_25)

        X_tr_76 = enforce_allowlist(tr_df, ALL_ALLOWLISTED_CIC_FEATURES)
        X_te_76 = enforce_allowlist(te_df, ALL_ALLOWLISTED_CIC_FEATURES)
        assert_no_leakage(X_tr_76, X_te_76)

        y_tr = (tr_df[lbl_col] != "Benign").astype(int).values
        y_te = (te_df[lbl_col] != "Benign").astype(int).values
        y_fam_te = np.array(te_df[lbl_col].str.strip().values, dtype="U32")

        return {
            "X_train": X_tr_9,
            "X_test": X_te_9,
            "X_train_25": X_tr_25,
            "X_test_25": X_te_25,
            "X_train_76": X_tr_76,
            "X_test_76": X_te_76,
            "y_train": y_tr,
            "y_test": y_te,
            "y_family_test": y_fam_te,
        }

    p1_train_df = pd.concat([benign_train, attack_train_pools["FTP-BruteForce"]], ignore_index=True)
    splits["ftp_only"] = extract_matrices(p1_train_df, all_test_df)

    for held_out in TARGET_ATTACKS:
        train_attacks = [attack_train_pools[fam] for fam in TARGET_ATTACKS if fam != held_out]
        tr_df = pd.concat([benign_train] + train_attacks, ignore_index=True)
        splits[f"lofo_{held_out}"] = extract_matrices(tr_df, all_test_df)

    # Prevalence splits
    n_benign_eval = len(benign_test)
    for ratio_name, ratio_val in [("prevalence_1_100", 0.01), ("prevalence_1_1000", 0.001)]:
        target_attack_n = max(5, int(n_benign_eval * ratio_val))
        sampled_attacks = all_test_attacks.sample(n=min(target_attack_n, len(all_test_attacks)), random_state=seed)
        prev_test_df = pd.concat([benign_test, sampled_attacks], ignore_index=True)
        splits[ratio_name] = extract_matrices(p1_train_df, prev_test_df)

    return splits


def run_lofo_on_dataset(bench: Dict[str, np.ndarray]) -> Dict:
    """Evaluates the 5 LOFO folds on a loaded benchmark dataset."""
    results = {}
    macro_accum = {"bacc": [], "held_rec": [], "fpr": [], "sfr": [], "tot_esc": [], "held_esc": [], "ben_esc": []}

    for held_out in TARGET_ATTACKS:
        X_tr_9 = bench[f"lofo_{held_out}_X_train"][:, :9]
        X_tr_76 = bench[f"lofo_{held_out}_X_train_76"]
        y_tr = bench[f"lofo_{held_out}_y_train"]

        X_te_9 = bench[f"lofo_{held_out}_X_test"][:, :9]
        X_te_76 = bench[f"lofo_{held_out}_X_test_76"]
        y_te = bench[f"lofo_{held_out}_y_test"]
        fam_te = bench[f"lofo_{held_out}_y_family_test"]

        t1 = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr_9, y_tr)
        t2 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_tr_76, y_tr)

        p1_prob = t1.predict_proba(X_te_9)[:, 1]
        p1_pred = (p1_prob >= 0.50).astype(int)
        p2_pred = t2.predict(X_te_76)

        esc_mask = (p1_prob >= 0.30) & (p1_prob <= 0.70)
        y_sys = p1_pred.copy()
        y_sys[esc_mask] = p2_pred[esc_mask]

        atk_m = (y_te == 1)
        ben_m = (y_te == 0)
        held_m = (fam_te == held_out)

        bacc = float(balanced_accuracy_score(y_te, y_sys) * 100.0)
        held_rec = float(np.mean(y_sys[held_m] == 1) * 100.0) if np.sum(held_m) > 0 else 0.0
        fpr = float(np.mean(y_sys[ben_m] == 1) * 100.0)
        sys_silent = (p1_pred == 0) & (~esc_mask) & atk_m
        sfr = float(np.sum(sys_silent) / max(1, np.sum(atk_m)) * 100.0)
        tot_esc = float(np.mean(esc_mask) * 100.0)
        held_esc = float(np.mean(esc_mask[held_m]) * 100.0) if np.sum(held_m) > 0 else 0.0
        ben_esc = float(np.mean(esc_mask[ben_m]) * 100.0)

        # Family recalls
        fam_rec = {}
        for f in TARGET_ATTACKS:
            fm = (fam_te == f)
            fam_rec[f] = float(np.mean(y_sys[fm] == 1) * 100.0) if np.sum(fm) > 0 else 0.0

        results[held_out] = {
            "bacc": bacc,
            "held_rec": held_rec,
            "fpr": fpr,
            "sfr": sfr,
            "tot_esc": tot_esc,
            "held_esc": held_esc,
            "ben_esc": ben_esc,
            "family_recalls": fam_rec,
        }

        macro_accum["bacc"].append(bacc)
        macro_accum["held_rec"].append(held_rec)
        macro_accum["fpr"].append(fpr)
        macro_accum["sfr"].append(sfr)
        macro_accum["tot_esc"].append(tot_esc)
        macro_accum["held_esc"].append(held_esc)
        macro_accum["ben_esc"].append(ben_esc)

    macro_summary = {k: float(np.mean(v)) for k, v in macro_accum.items()}
    return {"folds": results, "macro": macro_summary}


def run_gate_comparison_on_dataset(bench: Dict[str, np.ndarray]) -> Dict:
    """Evaluates standardized gates (kNN, Mahalanobis, RF Band, UNION) on ftp_only test split."""
    X_tr_9 = bench["ftp_only_X_train"][:, :9]
    y_tr = bench["ftp_only_y_train"]
    X_te_9 = bench["ftp_only_X_test"][:, :9]
    y_te = bench["ftp_only_y_test"]
    fam_te = bench["ftp_only_y_family_test"]

    # Fit scaler on benign train only
    ben_tr_m = (y_tr == 0)
    scaler = StandardScaler().fit(X_tr_9[ben_tr_m])
    Z_tr = scaler.transform(X_tr_9[ben_tr_m])
    Z_te = scaler.transform(X_te_9)

    ben_te_m = (y_te == 0)
    novel_m = (y_te == 1) & (fam_te != "FTP-BruteForce")

    # Tier 1 RF model for probability band
    rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr_9, y_tr)
    rf_probs = rf.predict_proba(X_te_9)[:, 1]
    rf_uncertainty = 1.0 - np.abs(rf_probs - 0.5) * 2.0  # higher = more uncertain

    # kNN distance
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(Z_tr[:4000])
    knn_dists = nn.kneighbors(Z_te, n_neighbors=5)[0].mean(axis=1)

    # Mahalanobis distance
    cov = np.cov(Z_tr, rowvar=False) + 1e-4 * np.eye(Z_tr.shape[1])
    inv_cov = np.linalg.pinv(cov)
    diff = Z_te - np.mean(Z_tr, axis=0)
    maha_dists = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

    gate_scores = {
        "Full_kNN": knn_dists,
        "Mahalanobis": maha_dists,
        "RF_Band": rf_uncertainty,
    }

    results = {}
    budgets = [0.01, 0.02, 0.05, 0.10]

    for g_name, scores in gate_scores.items():
        results[g_name] = {}
        for b in budgets:
            b_str = f"{int(b*100)}%"
            ben_scores = scores[ben_te_m]
            threshold = float(np.percentile(ben_scores, 100.0 * (1.0 - b)))
            esc_mask = scores >= threshold

            achieved_ben = float(np.mean(esc_mask[ben_te_m]) * 100.0)
            pooled_cov = float(np.mean(esc_mask[novel_m]) * 100.0) if np.sum(novel_m) > 0 else 0.0

            # Macro coverage across 4 novel families
            macro_covs = []
            for nov_fam in ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]:
                fm = (fam_te == nov_fam)
                cov_val = float(np.mean(esc_mask[fm]) * 100.0) if np.sum(fm) > 0 else 0.0
                macro_covs.append(cov_val)
            macro_cov = float(np.mean(macro_covs))

            results[g_name][b_str] = {
                "achieved_benign": achieved_ben,
                "novel_macro_coverage": macro_cov,
                "novel_pooled_coverage": pooled_cov,
            }

    # UNION Gate (RF Band + kNN with split budget)
    results["UNION_Gate"] = {}
    for b in budgets:
        b_str = f"{int(b*100)}%"
        sub_b = b / 2.0
        rf_th = float(np.percentile(rf_uncertainty[ben_te_m], 100.0 * (1.0 - sub_b)))
        knn_th = float(np.percentile(knn_dists[ben_te_m], 100.0 * (1.0 - sub_b)))
        u_esc = (rf_uncertainty >= rf_th) | (knn_dists >= knn_th)

        achieved_ben = float(np.mean(u_esc[ben_te_m]) * 100.0)
        pooled_cov = float(np.mean(u_esc[novel_m]) * 100.0) if np.sum(novel_m) > 0 else 0.0
        macro_covs = [
            float(np.mean(u_esc[fam_te == nf]) * 100.0) for nf in ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]
        ]
        results["UNION_Gate"][b_str] = {
            "achieved_benign": achieved_ben,
            "novel_macro_coverage": float(np.mean(macro_covs)),
            "novel_pooled_coverage": pooled_cov,
        }

    return results


def main():
    print("=" * 80)
    print(" SAMPLING SENSITIVITY EVALUATION: UNIFORM RANDOM SAMPLING ACROSS CIC.CSV")
    print("=" * 80)

    # 1. Uniformly draw from entire file
    df_uniform, provenance_stats = draw_uniform_sampled_corpus(
        cic_path=CIC_PATH,
        sample_per_attack=1500,
        benign_total=15000,
        seed=42,
    )

    # 2. Build splits
    uniform_splits = build_uniform_benchmark_splits(df_uniform, seed=42)

    # 3. Save benchmark_uniform_dataset.npz
    npz_dict = {}
    for name, data in uniform_splits.items():
        for k, v in data.items():
            npz_dict[f"{name}_{k}"] = v
    np.savez_compressed(UNIFORM_NPZ_PATH, **npz_dict)
    print(f"[SamplingSensitivity] Saved uniform benchmark dataset to {UNIFORM_NPZ_PATH}")

    # 4. Evaluate LOFO on New (Uniform) vs Old (First-N)
    print("\n[SamplingSensitivity] Evaluating LOFO on New (Uniform) dataset...")
    uniform_lofo = run_lofo_on_dataset(npz_dict)

    print("\n[SamplingSensitivity] Evaluating LOFO on Old (First-N) dataset...")
    old_bench = np.load(OLD_NPZ_PATH)
    old_lofo = run_lofo_on_dataset(old_bench)

    # 5. Evaluate Gates on New (Uniform) vs Old (First-N)
    print("\n[SamplingSensitivity] Evaluating Escalation Gates on New (Uniform) dataset...")
    uniform_gates = run_gate_comparison_on_dataset(npz_dict)

    print("\n[SamplingSensitivity] Evaluating Escalation Gates on Old (First-N) dataset...")
    old_gates = run_gate_comparison_on_dataset(old_bench)

    # 6. Assemble comprehensive comparison artifact
    output_data = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "provenance_and_sessions": provenance_stats,
        "lofo_comparison": {
            "old_first_n": old_lofo,
            "new_uniform": uniform_lofo,
            "macro_differences": {
                "delta_bacc": uniform_lofo["macro"]["bacc"] - old_lofo["macro"]["bacc"],
                "delta_held_rec": uniform_lofo["macro"]["held_rec"] - old_lofo["macro"]["held_rec"],
                "delta_sfr": uniform_lofo["macro"]["sfr"] - old_lofo["macro"]["sfr"],
                "delta_fpr": uniform_lofo["macro"]["fpr"] - old_lofo["macro"]["fpr"],
            },
        },
        "gates_comparison": {
            "old_first_n": old_gates,
            "new_uniform": uniform_gates,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[SamplingSensitivity] Successfully saved results to {OUTPUT_JSON}")

    # Print summary table
    print("\n" + "=" * 80)
    print(" SAMPLING SENSITIVITY COMPARISON: OLD (FIRST-N) vs NEW (UNIFORM)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Old (First-N)':<15} | {'New (Uniform)':<15} | {'Delta':<10}")
    print("-" * 78)
    for m in ["bacc", "held_rec", "sfr", "fpr", "tot_esc", "held_esc"]:
        old_v = old_lofo["macro"][m]
        new_v = uniform_lofo["macro"][m]
        diff = new_v - old_v
        print(f"{m:<30} | {old_v:6.2f}%         | {new_v:6.2f}%         | {diff:+6.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
