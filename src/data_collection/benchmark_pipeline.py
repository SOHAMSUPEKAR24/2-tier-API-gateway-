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
    wanted_cols = set(TIER1_ALLOWLIST + ["Label"])
    collected = {att: [] for att in target_attacks}
    collected["Benign"] = []

    print(f"[BenchmarkPipeline] Scanning {cic_path} with fast usecols...")
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

    # Clean: drop duplicates on allowlist to strictly prevent data leakage
    initial_len = len(df)
    df = df.drop_duplicates(subset=TIER1_ALLOWLIST).reset_index(drop=True)
    print(f"[BenchmarkPipeline] Deduplicated: {initial_len} -> {len(df)} unique flow profiles.")

    # Clean: inf / nan
    for col in TIER1_ALLOWLIST:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TIER1_ALLOWLIST] = df[TIER1_ALLOWLIST].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def build_protocol_splits(
    df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Constructs standardized benchmark splits:
      1. 'ftp_only': Train on FTP-BruteForce + Benign, Test on 4 novel families + Benign.
      2. 'lofo_<family>': Leave-one-family-out for each attack family.
      3. 'prevalence_1_100': Novel attack prevalence 1:100 against benign.
      4. 'prevalence_1_1000': Novel attack prevalence 1:1000 against benign.
    """
    lbl_col = [c for c in df.columns if "label" in c.lower()][0]
    families = ["FTP-BruteForce", "SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]

    splits = {}

    # Separate benign and attack pools
    benign_df = df[df[lbl_col] == "Benign"].reset_index(drop=True)

    # Chronological time-based split: early benign for train, late benign for test
    split_point = int(len(benign_df) * 0.60)
    benign_train = benign_df.iloc[:split_point]
    benign_test = benign_df.iloc[split_point:]

    # ── Protocol 1: Train FTP only, test on 4 novel families ─────────────────
    ftp_df = df[df[lbl_col] == "FTP-BruteForce"].reset_index(drop=True)
    ftp_train_n = int(len(ftp_df) * 0.70)
    p1_train_df = pd.concat([benign_train, ftp_df.iloc[:ftp_train_n]], ignore_index=True)

    novel_test_dfs = [ftp_df.iloc[ftp_train_n:]]
    for f in ["SSH-BruteForce", "DoS-GoldenEye", "DoS-Hulk", "DoS-Slowloris"]:
        fam_df = df[df[lbl_col] == f]
        novel_test_dfs.append(fam_df)

    p1_test_attacks = pd.concat(novel_test_dfs, ignore_index=True)
    p1_test_df = pd.concat([benign_test, p1_test_attacks], ignore_index=True)

    X_tr_p1 = enforce_allowlist(p1_train_df, TIER1_ALLOWLIST)
    y_tr_p1 = (p1_train_df[lbl_col] != "Benign").astype(int).values
    X_te_p1 = enforce_allowlist(p1_test_df, TIER1_ALLOWLIST)
    y_te_p1 = (p1_test_df[lbl_col] != "Benign").astype(int).values

    # Check leakage
    assert_no_leakage(X_tr_p1, X_te_p1)

    splits["ftp_only"] = {
        "X_train": X_tr_p1,
        "y_train": y_tr_p1,
        "X_test": X_te_p1,
        "y_test": y_te_p1,
    }

    # ── Protocol 2: Leave-One-Family-Out (LOFO) ──────────────────────────────
    for held_out in families:
        train_attacks = [df[df[lbl_col] == fam] for fam in families if fam != held_out]
        train_attacks_df = pd.concat(train_attacks, ignore_index=True)
        test_attacks_df = df[df[lbl_col] == held_out].reset_index(drop=True)

        tr_df = pd.concat([benign_train, train_attacks_df], ignore_index=True)
        te_df = pd.concat([benign_test, test_attacks_df], ignore_index=True)

        X_tr = enforce_allowlist(tr_df, TIER1_ALLOWLIST)
        y_tr = (tr_df[lbl_col] != "Benign").astype(int).values
        X_te = enforce_allowlist(te_df, TIER1_ALLOWLIST)
        y_te = (te_df[lbl_col] != "Benign").astype(int).values

        assert_no_leakage(X_tr, X_te)

        splits[f"lofo_{held_out}"] = {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
        }

    # ── Protocol 3: Realistic Prevalence (1:100 and 1:1000) ──────────────────
    n_benign_eval = len(benign_test)
    for ratio_name, ratio_val in [("prevalence_1_100", 0.01), ("prevalence_1_1000", 0.001)]:
        target_attack_n = max(5, int(n_benign_eval * ratio_val))
        sampled_attacks = p1_test_attacks.sample(n=min(target_attack_n, len(p1_test_attacks)), random_state=seed)
        prev_test_df = pd.concat([benign_test, sampled_attacks], ignore_index=True)

        X_prev = enforce_allowlist(prev_test_df, TIER1_ALLOWLIST)
        y_prev = (prev_test_df[lbl_col] != "Benign").astype(int).values

        assert_no_leakage(X_tr_p1, X_prev)

        splits[ratio_name] = {
            "X_train": X_tr_p1,
            "y_train": y_tr_p1,
            "X_test": X_prev,
            "y_test": y_prev,
        }

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
        "features": TIER1_ALLOWLIST,
        "feature_count": len(TIER1_ALLOWLIST),
        "total_unique_flows": len(df),
        "protocols": {},
    }

    for name, data in splits.items():
        npz_dict[f"{name}_X_train"] = data["X_train"]
        npz_dict[f"{name}_y_train"] = data["y_train"]
        npz_dict[f"{name}_X_test"] = data["X_test"]
        npz_dict[f"{name}_y_test"] = data["y_test"]

        stats_dict["protocols"][name] = {
            "train_samples": int(len(data["y_train"])),
            "train_attacks": int(np.sum(data["y_train"])),
            "train_benign": int(np.sum(data["y_train"] == 0)),
            "test_samples": int(len(data["y_test"])),
            "test_attacks": int(np.sum(data["y_test"])),
            "test_benign": int(np.sum(data["y_test"] == 0)),
            "attack_prevalence_pct": float(np.mean(data["y_test"]) * 100),
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
