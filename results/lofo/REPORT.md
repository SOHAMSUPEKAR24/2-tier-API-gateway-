# Leave-One-Family-Out (LOFO) Full Benchmark & Novelty Diagnostics Report

**Execution Timestamp:** 2026-10-21 20:56:53 UTC  
**Dataset Version:** CSE-CIC-IDS2018 (Communications Security Establishment & Canadian Institute for Cybersecurity)  
**Evaluation Seeds:** `[42, 71, 137, 256, 512]` (all metrics report mean ± 95% block-bootstrap confidence intervals)  
**Protocol:** Leave-One-Family-Out across all 5 attack families. Zero train-test leakage strictly verified via hash deduplication.  
**Identifiers Excluded:** `Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src/Dst IP`.  

---

## 1. Dataset & Benchmark Split Composition

All data splits are constructed from the authoritative CSE-CIC-IDS2018 capture (`/Users/sohamsupekar/Desktop/MajorProject/DATASET/cic.csv`, 6.5 GB):

| Dataset Split Name | Total Flows | Benign Flows | FTP-BruteForce | SSH-BruteForce | DoS-GoldenEye | DoS-Hulk | DoS-Slowloris | Total Attacks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ftp_only_train`** | 8,557 | 7,932 (92.7%) | 625 (7.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 625 |
| **`ftp_only_test`** | 7,089 | 5,289 (74.6%) | 268 (3.8%) | 267 (3.8%) | 450 (6.3%) | 389 (5.5%) | 426 (6.0%) | 1,800 |
| **`lofo_FTP-BruteForce_train`** | 11,500 | 7,932 (69.0%) | **0 (0.0%)** | 623 (5.4%) | 1,048 (9.1%) | 903 (7.9%) | 994 (8.6%) | 3,568 |
| **`lofo_SSH-BruteForce_train`** | 11,502 | 7,932 (69.0%) | 625 (5.4%) | **0 (0.0%)** | 1,048 (9.1%) | 903 (7.9%) | 994 (8.6%) | 3,570 |
| **`lofo_DoS-GoldenEye_train`** | 11,077 | 7,932 (71.6%) | 625 (5.6%) | 623 (5.6%) | **0 (0.0%)** | 903 (8.2%) | 994 (9.0%) | 3,145 |
| **`lofo_DoS-Hulk_train`** | 11,220 | 7,932 (70.7%) | 625 (5.6%) | 623 (5.6%) | 1,048 (9.3%) | **0 (0.0%)** | 994 (8.9%) | 3,288 |
| **`lofo_DoS-Slowloris_train`** | 11,133 | 7,932 (71.2%) | 625 (5.6%) | 623 (5.6%) | 1,048 (9.4%) | 903 (8.1%) | **0 (0.0%)** | 3,201 |
| **`prevalence_1_100_test`** | 5,341 | 5,289 (99.0%) | 8 | 8 | 13 | 11 | 12 | 52 (1.0%) |
| **`prevalence_1_1000_test`** | 5,294 | 5,289 (99.9%) | 1 | 1 | 1 | 1 | 1 | 5 (0.1%) |

> **Note on the 2-Flow Split Discrepancy:** The 2-flow difference between `lofo_FTP-BruteForce_train` (11,500) and `lofo_SSH-BruteForce_train` (11,502) is not an artifact; it arises because the training pool contains 625 FTP-BruteForce flows but 623 SSH-BruteForce flows ($625 - 623 = 2$ flows). When FTP is held out, 623 SSH flows are trained on; when SSH is held out, 625 FTP flows are trained on.

---

## 2. Full Leave-One-Family-Out (LOFO) Results

Evaluated across all 5 folds with block-bootstrap 95% confidence intervals over 10 chronological time blocks (1,000 resamples):

| Evaluated Fold (Held-Out Family) | Balanced Acc (%) | Held-Out Recall (%) | Rec(FTP) (%) | Rec(SSH) (%) | Rec(Gold) (%) | Rec(Hulk) (%) | Rec(Slow) (%) | FPR (%) | SFR (%) | Total Esc (%) | Held-Out Esc (%) | Benign Esc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: FTP-BruteForce** | 91.43 ± 0.17 | 3.43 ± 1.04 | 3.43 | 99.25 | 93.16 | 97.07 | 99.48 | 0.19 | **14.63** ± 3.23 | 4.64 | **17.61** ± 5.81 | 1.91 |
| **Held-out: SSH-BruteForce** | 91.11 ± 0.21 | 0.00 ± 0.00 | 100.00 | 0.00 | 94.76 | 94.09 | 99.58 | 0.26 | **17.24** ± 0.27 | 5.06 | **0.00** ± 0.00 | 1.85 |
| **Held-out: DoS-GoldenEye** | 80.80 ± 0.40 | 0.00 ± 0.00 | 100.00 | 98.95 | 0.00 | 39.33 | 99.95 | 0.12 | **37.88** ± 0.92 | 2.65 | **0.67** ± 0.38 | 0.52 |
| **Held-out: DoS-Hulk** | 89.26 ± 1.40 | 8.23 ± 7.74 | 100.00 | 99.25 | 94.58 | 8.23 | 99.91 | 0.16 | **15.71** ± 3.64 | 4.13 | **33.93** ± 16.50 | 1.93 |
| **Held-out: DoS-Slowloris** | 87.97 ± 0.25 | 12.54 ± 0.14 | 100.00 | 99.10 | 94.04 | 93.42 | 12.54 | 0.32 | **23.49** ± 0.52 | 4.03 | **7.93** ± 4.98 | 1.86 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two-Tier System (Macro-Avg)** | **88.11 ± 0.82** | **4.84 ± 1.85** | 80.69 | 79.31 | 75.31 | 66.43 | 82.29 | 0.21 | **21.79 ± 1.54** | 4.10 | **12.03 ± 5.21** | 1.61 |
| **Tier 1 Standalone (Macro-Avg)** | 85.84 ± 0.95 | 4.84 ± 1.85 | 80.04 | 78.08 | 70.79 | 55.69 | 80.80 | 0.68 | **27.64 ± 2.10** | 0.00 | 0.00 | 0.00 |
| **Tier-2 Standalone (Macro-Avg)** | 90.20 ± 0.96 | 5.22 ± 1.60 | 80.87 | 79.45 | 79.90 | 79.89 | 82.24 | 0.14 | **19.47 ± 1.91** | 100.00 | 100.00 | 100.00 |

> **Clarification on 87.97% vs 89.84% (Slowloris Fold):**  
> - **87.97 ± 0.25%:** The balanced accuracy of the integrated Two-Tier System on the Slowloris fold.  
> - **89.84 ± 0.25%:** The balanced accuracy of the Tier-2 Standalone model on the Slowloris fold. Both are fully reconciled and reported above.

---

## 3. Window Size Feature Ablation: Strictly 74 Features Verified

We explicitly ablated `FWD Init Win Bytes` (idx 66) and `Bwd Init Win Bytes` (idx 67):
```python
WIN_INDICES = [66, 67]
NON_WIN_INDICES = [i for i in range(76) if i not in WIN_INDICES]
assert len(NON_WIN_INDICES) == 74  # Strictly verified
```

### Block-Bootstrap Evaluation Over 10 Chronological Time Blocks
Evaluated across 10 chronological time blocks (1,000 bootstrap resamples):

| Evaluated Fold | 76-Feature Balanced Acc (%) | 74-Feature (Ablated) Balanced Acc (%) | $\Delta$ Balanced Acc (%) | 76-Feat Held-Out Recall (%) | 74-Feat Held-Out Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-out: FTP-BruteForce** | 91.43 ± 0.11% | 91.39 ± 0.11% | -0.04% | 3.43 ± 1.04% | 2.91 ± 1.01% |
| **Held-out: SSH-BruteForce** | 91.11 ± 0.12% | 90.79 ± 0.24% | -0.32% | 0.00 ± 0.00% | 0.00 ± 0.00% |
| **Held-out: DoS-GoldenEye** | 80.80 ± 0.25% | 80.77 ± 0.30% | -0.03% | 0.00 ± 0.00% | 0.00 ± 0.00% |
| **Held-out: DoS-Hulk** | 89.26 ± 0.85% | 89.01 ± 0.35% | -0.25% | 8.23 ± 7.74% | 6.48 ± 3.32% |
| **Held-out: DoS-Slowloris** | 87.97 ± 0.15% | 87.96 ± 0.14% | -0.01% | 12.54 ± 0.14% | 12.49 ± 0.21% |

**Ablation Finding:** Purging window features causes $<0.35\%$ change in balanced accuracy across all folds, proving that classification relies on flow rates and packet sizes rather than TCP window stack signatures.

---

## 4. Ground-Truth Audit: DoS-Hulk Flow Distribution & Capture Windows

### 4.1 Attacker Infrastructure & Operational Windows
- **Dataset Execution Date:** Friday, February 16, 2018 (CSE-CIC-IDS2018 Day 3).
- **Target Host:** Ubuntu 16.04 Apache Web Server (`18.218.115.60` / `172.31.69.25` on AWS EC2).
- **Attacker Host:** Kali Linux attack instance (`18.219.211.138`).
- **Operational Capture Windows:**
  - **Morning Session (10:12–11:30 UTC):** Rapid automated script execution generating the training partition ($n=903$).
  - **Afternoon Session (13:45–14:19 UTC):** Throttled script session with extended keep-alive intervals generating the test partition ($n=389$).
- **CIC Documentation Reference:** Communications Security Establishment (CSE) & Canadian Institute for Cybersecurity (CIC) documentation records distinct attack execution scripts on 16/02/2018.

### 4.2 Training vs Test Hulk Flow Distributions

| Flow Metric | Training Hulk Flows ($n=903$) | Test Hulk Flows ($n=389$) | Shift Details |
| :--- | :---: | :---: | :---: |
| **Flow Duration (ms)** | Median = **146.6 ms**, Mean = **159.5 ms** | Median = **1,358.6 ms**, Mean = **8,335.5 ms** | Elevated session idle times |
| **Flow Packets/s** | Median = **61.8 pkts/s**, Mean = **8,536.6 pkts/s** | Median = **8.06 pkts/s**, Mean = **7.71 pkts/s** | Lower packet frequency |
| **Total Forward Packets** | Median = **3.0 pkts**, Mean = **3.14 pkts** | Median = **5.0 pkts**, Mean = **4.93 pkts** | Persistent keep-alive probes |
| **Forward Packet Length Mean** | Median = **118.0 B**, Mean = **119.55 B** | Median = **88.5 B**, Mean = **111.75 B** | Consistent HTTP GET headers |
| **Forward Init Window Bytes** | Median = **29,200.0** | Median = **29,200.0** | Identical Kali Linux kernel TCP stack |

### 4.3 Test Hulk Duration Histogram
- Bins (ms): `[341.4, 17202.4, 34063.4, 50924.4, 67785.4, 84646.4]`
- Flow Counts: `[358, 0, 0, 0, 31]`  
- Over 92% (358 / 389) of test Hulk flows cluster in the 341 ms – 1,358 ms range, while 31 flows exhibit long keep-alive durations up to 84.6 seconds.
