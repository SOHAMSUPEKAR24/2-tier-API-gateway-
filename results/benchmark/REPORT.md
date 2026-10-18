# Phase 2 Benchmark Report: Reproducible Open-Set Benchmark

**Date:** 2026-10-18  
**Dataset:** CSE-CIC-IDS2018 (Cleaned, CICFlowMeter-V3)  
**Artifacts Generated:**  
- `scripts/download_datasets.sh` (Acquisition pipeline for CSE-CIC-IDS2018, CIC-IDS2017, CSIC 2010)
- `src/data_collection/benchmark_pipeline.py` (Deduplication, feature guarding, protocol splits)
- `tests/test_benchmark_leakage.py` (5 passing leakage & feature guard tests)
- `data/processed/benchmark_dataset.npz` (Compressed pre-split benchmark dataset)
- `results/benchmark_stats.json` (Exact split sample sizes and prevalence stats)

---

## 1. What Was Run & Cleaning Methodology
1. **Cleaning & Feature Guard:**
   - Evaluated 22,500 candidate flows across 5 attack families (`FTP-BruteForce`, `SSH-BruteForce`, `DoS-GoldenEye`, `DoS-Hulk`, `DoS-Slowloris`) and Benign background traffic.
   - Strictly enforced the 9-feature `TIER1_ALLOWLIST` (Fwd/Bwd Packet Lengths, Flow Bytes/s, Flow Packets/s, IAT, Flags).
   - Removed all identifier columns (`Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src/Dst IP`).
   - Deduplicated on allowlist features: 22,500 -> **19,214 unique flow profiles**.
   - Filled non-finite flow rates (+/- inf) with 0.0.

2. **Benchmark Protocols Implemented:**
   - **Protocol A ("FTP Only"):** Train on early Benign + FTP-BruteForce (8,557 samples); test on held-out Benign + 4 novel attack families (`SSH-BruteForce`, `DoS-GoldenEye`, `DoS-Hulk`, `DoS-Slowloris`, 10,657 samples).
   - **Protocol B (Leave-One-Family-Out / LOFO):** 5 separate configurations where each attack family is held out as novel while the model trains on the remaining 4 families + Benign.
   - **Protocol C (Realistic Prevalence):** Subsampled attack streams to 1:100 (0.97% attack prevalence) and 1:1000 (0.094% attack prevalence) against 5,289 held-out benign flows to model edge gateway base-rate fallacy dynamics.

3. **Leakage Verification:**
   - Tested using `tests/test_benchmark_leakage.py` with `pytest`.
   - Guaranteed **0 duplicate rows** between train and test splits across all protocols.

---

## 2. Benchmark Split Statistics Summary

| Protocol | Train Samples (Benign / Attack) | Test Samples (Benign / Attack) | Attack Prevalence in Test |
| :--- | :--- | :--- | :--- |
| **`ftp_only`** | 8,557 (7,932 / 625) | 10,657 (5,289 / 5,368) | 50.37% |
| **`lofo_FTP-BruteForce`** | 13,032 (7,932 / 5,100) | 6,182 (5,289 / 893) | 14.45% |
| **`lofo_SSH-BruteForce`** | 13,035 (7,932 / 5,103) | 6,179 (5,289 / 890) | 14.40% |
| **`lofo_DoS-GoldenEye`** | 12,427 (7,932 / 4,495) | 6,787 (5,289 / 1,498) | 22.07% |
| **`lofo_DoS-Hulk`** | 12,631 (7,932 / 4,699) | 6,583 (5,289 / 1,294) | 19.66% |
| **`lofo_DoS-Slowloris`** | 12,507 (7,932 / 4,575) | 6,707 (5,289 / 1,418) | 21.14% |
| **`prevalence_1_100`** | 8,557 (7,932 / 625) | 5,341 (5,289 / 52) | 0.97% |
| **`prevalence_1_1000`** | 8,557 (7,932 / 625) | 5,294 (5,289 / 5) | 0.094% |

---

## 3. Known Limitations & Documented Label Issues
- **CICFlowMeter Zero-Length Timeouts:** Flows with duration zero occasionally emit `inf` in bytes/second or packet/second calculations; replaced with 0.0.
- **Port Inversion in High Load:** During DoS-Hulk floods, client ephemeral ports and server ports can reverse in raw PCAP reassembly; dropping `Dst Port` and all port features protects our models against this known dataset error.
