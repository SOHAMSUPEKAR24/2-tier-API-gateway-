# Phase 2 Benchmark Report: Reproducible Open-Set Benchmark

**Date:** 2026-10-21 21:40:56 UTC  
**Dataset:** CSE-CIC-IDS2018 (Cleaned, CICFlowMeter-V3)  
**Artifacts Generated:**  
- `scripts/download_datasets.sh` (Acquisition pipeline for CSE-CIC-IDS2018, CIC-IDS2017, CSIC 2010)
- `src/data_collection/benchmark_pipeline.py` (Deduplication, feature guarding, protocol splits)
- `tests/test_benchmark_leakage.py` (Passing leakage & feature guard tests)
- `data/processed/benchmark_dataset.npz` (Compressed pre-split benchmark dataset)
- `results/benchmark_stats.json` (Exact split sample sizes and prevalence stats)

---

## 1. What Was Run & Cleaning Methodology
1. **Cleaning & Feature Guard:**
   - Evaluated candidate flows across 5 attack families (`FTP-BruteForce`, `SSH-BruteForce`, `DoS-GoldenEye`, `DoS-Hulk`, `DoS-Slowloris`) and Benign background traffic.
   - Strictly enforced allowlisted features (Fwd/Bwd Packet Lengths, Flow Bytes/s, Flow Packets/s, IAT, Flags).
   - Removed all identifier columns (`Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src/Dst IP`).
   - Deduplicated on allowlist features.
   - Filled non-finite flow rates with 0.0.

2. **Benchmark Protocols Implemented:**
   - **Protocol A ("FTP Only"):** Train on early Benign + FTP-BruteForce; test on held-out Benign + 4 novel attack families (`SSH-BruteForce`, `DoS-GoldenEye`, `DoS-Hulk`, `DoS-Slowloris`).
   - **Protocol B (Leave-One-Family-Out / LOFO):** 5 separate configurations where each attack family is held out as novel while the model trains on the remaining 4 families + Benign.
   - **Protocol C (Realistic Prevalence):** Subsampled attack streams to 1:100 and 1:1000 against held-out benign flows to model edge gateway base-rate fallacy dynamics.

3. **Leakage Verification:**
   - Guaranteed **0 duplicate rows** between train and test splits across all protocols.

---

## 2. Benchmark Split Statistics Summary

| Protocol | Train Samples (Benign / Attack) | Test Samples (Benign / Attack) | Attack Prevalence in Test |
| :--- | :--- | :--- | :--- |
| **`ftp_only`** | 8,557 (7,932 / 625) | 7,089 (5,289 / 1,800) | 25.39% |
| **`lofo_FTP-BruteForce`** | 11,500 (7,932 / 3,568) | 7,089 (5,289 / 1,800) | 25.39% |
| **`lofo_SSH-BruteForce`** | 11,502 (7,932 / 3,570) | 7,089 (5,289 / 1,800) | 25.39% |
| **`lofo_DoS-GoldenEye`** | 11,077 (7,932 / 3,145) | 7,089 (5,289 / 1,800) | 25.39% |
| **`lofo_DoS-Hulk`** | 11,220 (7,932 / 3,288) | 7,089 (5,289 / 1,800) | 25.39% |
| **`lofo_DoS-Slowloris`** | 11,133 (7,932 / 3,201) | 7,089 (5,289 / 1,800) | 25.39% |
| **`prevalence_1_100`** | 8,557 (7,932 / 625) | 5,341 (5,289 / 52) | 0.97% |
| **`prevalence_1_1000`** | 8,557 (7,932 / 625) | 5,294 (5,289 / 5) | 0.094% |

---

## 3. Known Limitations & Documented Label Issues
- **CICFlowMeter Zero-Length Timeouts:** Flows with duration zero occasionally emit `inf` in bytes/second or packet/second calculations; replaced with 0.0.
- **Port Inversion in High Load:** During DoS-Hulk floods, client ephemeral ports and server ports can reverse in raw PCAP reassembly; dropping `Dst Port` and all port features protects our models against this known dataset error.
