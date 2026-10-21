"""
src/data_collection/benchmark_pipeline.py
──────────────────────────────────────────
Authoritative Data Cleaning and Open-Set Benchmark Split Generator.
Complies with Project Rules 1, 3, 4:
  - No identifier columns (enforces allowlist at read time)
  - Zero train-test leakage (strictly deduplicated across allowlist)
  - Seeded, time-based splits
  - Leave-One-Family-Out (LOFO) and Prevalence (1:100, 1:1000) protocols
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.feature_guard import (
    ALL_ALLOWLISTED_CIC_FEATURES,
    FEATURES_9,
    FEATURES_25,
    IDENTIFIER_COLUMNS,
    TIER1_ALLOWLIST,
    assert_no_leakage,
    enforce_allowlist,
    strip_identifiers,
)

CIC_PATH = "/Users/sohamsupekar/Desktop/MajorProject/DATASET/cic.csv"
BENCHMARK_PROCESSED_PATH = "data/processed/benchmark_dataset.npz"

KNOWN_LABEL_ISSUES = (
    "CSE-CIC-IDS2018 and CIC-IDS2017 contain documented label noise, including "
    "flow direction inversion on high-load DoS runs, short-duration timeout artifacts, "
    "and duplicate flows from mirrored switch ports. Our pipeline drops all duplicate rows "
    "on allowlist features, cleans non-finite flow rates, and excludes identifier columns "
    "(IP, Port, Timestamp, Flow ID) to eliminate trivial spurious correlations."
)


def extract_clean_raw_corpus(
    cic_path: str = CIC_PATH,
    sample_per_attack: int = 1500,
    benign_total: int = 15000,
    max_chunks: int = 40,
) -> pd.DataFrame:
    """
    Extract a clean, deduplicated, multi-attack corpus directly from cic.csv in chunks.
    Reads ONLY allowlisted features + Label via usecols to avoid parsing identifier columns.
    Preserves chronological arrival order (time-based capture).
    """
    target_attacks = [
        "FTP-BruteForce",
        "SSH-BruteForce",
        "DoS-GoldenEye",
        "DoS-Hulk",
        "DoS-Slowloris",
    ]
    wanted_cols = set(ALL_ALLOWLISTED_CIC_FEATURES + ["Label"])
    collected = {att: [] for att in target_attacks}
    collected["Benign"] = []

    print(f"[BenchmarkPipeline] Scanning {cic_path} with fast usecols (76 allowlisted features)...")
    reader = pd.read_csv(
        cic_path,
        chunksize=100000,
        usecols=lambda c: c.strip() in wanted_cols,
    )

    for chunk_idx, chunk in enumerate(reader):
        chunk.columns = [c.strip() for c in chunk.columns]
        lbl_col = [c for c in chunk.columns if "label" in c.lower()][0]

        # Benign
        curr_b = sum(len(x) for x in collected["Benign"])
        if curr_b < benign_total:
            b = chunk[chunk[lbl_col] == "Benign"].head(benign_total - curr_b)
            if len(b) > 0:
                collected["Benign"].append(b)

        # Attacks
        for att in target_attacks:
            curr_a = sum(len(x) for x in collected[att])
            if curr_a < sample_per_attack:
                a = chunk[chunk[lbl_col].str.strip() == att].head(sample_per_attack - curr_a)
                if len(a) > 0:
                    collected[att].append(a)

        all_attacks_ready = all(
            sum(len(x) for x in collected[att]) >= sample_per_attack for att in target_attacks
        )
        curr_b_total = sum(len(x) for x in collected["Benign"])
        if all_attacks_ready and curr_b_total >= benign_total:
            print(f"[BenchmarkPipeline] Target quotas reached at chunk {chunk_idx}.")
            break

        if chunk_idx >= max_chunks:
            print(f"[BenchmarkPipeline] Reached max chunks ({max_chunks}).")
            break

    frames = []
    for k, v in collected.items():
        if v:
            f = pd.concat(v, ignore_index=True)
            frames.append(f)
            print(f"  Captured {k}: {len(f)} flows")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    # Clean: remove any stray identifiers
    df = strip_identifiers(df)

    # Clean: drop duplicates on FEATURES_9 to strictly prevent data leakage across all feature subsets
    initial_len = len(df)
    df = df.drop_duplicates(subset=FEATURES_9).reset_index(drop=True)
    print(f"[BenchmarkPipeline] Deduplicated: {initial_len} -> {len(df)} unique flow profiles.")

    # Clean: inf / nan across all 76 features
    for col in ALL_ALLOWLISTED_CIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[ALL_ALLOWLISTED_CIC_FEATURES] = df[ALL_ALLOWLISTED_CIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def build_protocol_splits(
    df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Constructs standardized benchmark splits with 9, 25, and 76 features, plus family labels:
      1. 'ftp_only': Train on FTP-BruteForce + Benign, Test on all 5 families + Benign.
      2. 'lofo_<family>': Leave-one-family-out for each attack family.
      3. 'prevalence_1_100': Novel attack prevalence 1:100 against benign.
      4. 'prevalence_1_1000': Novel attack prevalence 1:1000 against benign.
    """
    lbl_col = [c for c in df.columns if "label" in c.lower()][0]
    families = ["FTP-BruteForce", "SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]

    splits = {}

    # Separate benign pool chronologically: 60% train, 40% test
    benign_df = df[df[lbl_col] == "Benign"].reset_index(drop=True)
    b_split = int(len(benign_df) * 0.60)
    benign_train = benign_df.iloc[:b_split]
    benign_test = benign_df.iloc[b_split:]

    # Partition each attack family chronologically: 70% train pool, 30% test pool
    attack_train_pools = {}
    attack_test_pools = {}
    for fam in families:
        fam_df = df[df[lbl_col] == fam].reset_index(drop=True)
        split_idx = int(len(fam_df) * 0.70)
        attack_train_pools[fam] = fam_df.iloc[:split_idx]
        attack_test_pools[fam] = fam_df.iloc[split_idx:]

    # Canonical all-family test attacks (30% held-out test portion of all 5 families)
    all_test_attacks = pd.concat([attack_test_pools[fam] for fam in families], ignore_index=True)
    all_test_df = pd.concat([benign_test, all_test_attacks], ignore_index=True)

    def extract_split_matrices(tr_df: pd.DataFrame, te_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        # 9 features
        X_tr_9 = enforce_allowlist(tr_df, FEATURES_9)
        X_te_9 = enforce_allowlist(te_df, FEATURES_9)
        assert_no_leakage(X_tr_9, X_te_9)

        # 25 features
        X_tr_25 = enforce_allowlist(tr_df, FEATURES_25)
        X_te_25 = enforce_allowlist(te_df, FEATURES_25)

        # 76 features
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

    # ── Protocol 1: Train FTP only (Regime B: Cold-start), test on all families ──
    p1_train_df = pd.concat([benign_train, attack_train_pools["FTP-BruteForce"]], ignore_index=True)
    splits["ftp_only"] = extract_split_matrices(p1_train_df, all_test_df)

    # ── Protocol 2: Leave-One-Family-Out (LOFO) (Regime A) ────────────────────
    for held_out in families:
        # Train on benign + the other 4 families' train portions
        train_attacks = [attack_train_pools[fam] for fam in families if fam != held_out]
        tr_df = pd.concat([benign_train] + train_attacks, ignore_index=True)
        # Test on benign_test + all 5 families test portions (with held_out labeled in y_family_test)
        splits[f"lofo_{held_out}"] = extract_split_matrices(tr_df, all_test_df)

    # ── Protocol 3: Realistic Prevalence (1:100 and 1:1000) ──────────────────
    n_benign_eval = len(benign_test)
    for ratio_name, ratio_val in [("prevalence_1_100", 0.01), ("prevalence_1_1000", 0.001)]:
        target_attack_n = max(5, int(n_benign_eval * ratio_val))
        sampled_attacks = all_test_attacks.sample(n=min(target_attack_n, len(all_test_attacks)), random_state=seed)
        prev_test_df = pd.concat([benign_test, sampled_attacks], ignore_index=True)
        splits[ratio_name] = extract_split_matrices(p1_train_df, prev_test_df)

    return splits


def generate_benchmark_artifacts():
    """
    Full pipeline run: extracts clean corpus, builds splits, checks leakage,
    serializes clean splits, and writes results/benchmark_stats.json.
    """
    df = extract_clean_raw_corpus()
    splits = build_protocol_splits(df, seed=42)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    npz_dict = {}
    stats_dict = {
        "dataset_name": "CSE-CIC-IDS2018 Cleaned Open-Set Benchmark",
        "provenance": "Generated from cic.csv (processed traffic data for ML algorithms, CICFlowMeter-V3)",
        "known_label_issues": KNOWN_LABEL_ISSUES,
        "features": FEATURES_9,
        "feature_count": len(FEATURES_9),
        "features_25_count": len(FEATURES_25),
        "features_76_count": len(ALL_ALLOWLISTED_CIC_FEATURES),
        "total_unique_flows": len(df),
        "protocols": {},
    }

    for name, data in splits.items():
        for k, v in data.items():
            npz_dict[f"{name}_{k}"] = v

        stats_dict["protocols"][name] = {
            "train_samples": int(len(data["y_train"])),
            "train_attacks": int(np.sum(data["y_train"])),
            "train_benign": int(np.sum(data["y_train"] == 0)),
            "test_samples": int(len(data["y_test"])),
            "test_attacks": int(np.sum(data["y_test"])),
            "test_benign": int(np.sum(data["y_test"] == 0)),
            "attack_prevalence_pct": float(np.mean(data["y_test"]) * 100),
            "features_9_shape": list(data["X_train"].shape),
            "features_25_shape": list(data["X_train_25"].shape),
            "features_76_shape": list(data["X_train_76"].shape),
        }

    np.savez_compressed(BENCHMARK_PROCESSED_PATH, **npz_dict)
    print(f"[BenchmarkPipeline] Saved benchmark splits to {BENCHMARK_PROCESSED_PATH}")

    stats_file = Path("results/benchmark_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[BenchmarkPipeline] Wrote benchmark stats to {stats_file}")

    return stats_dict


if __name__ == "__main__":
    generate_benchmark_artifacts()

