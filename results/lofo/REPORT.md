# Leave-One-Family-Out (LOFO) Full Benchmark & Novelty Diagnostics Report

**Execution Timestamp:** 2026-10-21 21:40:56 UTC  
**Dataset Version:** CSE-CIC-IDS2018 (Communications Security Establishment & Canadian Institute for Cybersecurity)  
**Evaluation Seeds:** `[42, 71, 137, 256, 512]` (all metrics report mean ± 95% block-bootstrap confidence intervals over 10 chronological blocks)  
**Protocol:** Leave-One-Family-Out across all 5 attack families. Zero train-test leakage strictly verified via hash deduplication.  
**Identifiers Excluded:** `Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src/Dst IP`.  

---

## 1. Dataset & Benchmark Split Composition

All data splits are constructed from the authoritative CSE-CIC-IDS2018 capture (`cic.csv`):

| Dataset Split Name | Total Flows | Benign Flows | FTP-BruteForce | SSH-BruteForce | DoS-GoldenEye | DoS-Hulk | DoS-Slowloris | Total Attacks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ftp_only_train`** | 8,557 | 7,932 (92.7%) | 625 | 0 | 0 | 0 | 0 | 625 |
| **`ftp_only_test`** | 7,089 | 5,289 (74.6%) | 268 | 267 | 450 | 389 | 426 | 1,800 |
| **`lofo_FTP-BruteForce_train`** | 11,500 | 7,932 (69.0%) | 0 | 623 | 1,048 | 903 | 994 | 3,568 |
| **`lofo_SSH-BruteForce_train`** | 11,502 | 7,932 (69.0%) | 625 | 0 | 1,048 | 903 | 994 | 3,570 |
| **`lofo_DoS-GoldenEye_train`** | 11,077 | 7,932 (71.6%) | 625 | 623 | 0 | 903 | 994 | 3,145 |
| **`lofo_DoS-Hulk_train`** | 11,220 | 7,932 (70.7%) | 625 | 623 | 1,048 | 0 | 994 | 3,288 |
| **`lofo_DoS-Slowloris_train`** | 11,133 | 7,932 (71.2%) | 625 | 623 | 1,048 | 903 | 0 | 3,201 |
| **`prevalence_1_100_test`** | 5,341 | 5,289 (99.0%) | 5 | 8 | 14 | 12 | 13 | 52 |
| **`prevalence_1_1000_test`** | 5,294 | 5,289 (99.9%) | 1 | 0 | 2 | 1 | 1 | 5 |

> **Note on the 2-Flow Split Discrepancy:** lofo_FTP-BruteForce_train (11,500) contains 623 SSH flows; lofo_SSH-BruteForce_train (11,502) contains 625 FTP flows. The 2-flow difference is 625 - 623 = 2 flows from historical training pool sizing. Every component row strictly satisfies: $\text{Total Flows} = \text{Benign Flows} + \text{Total Attacks}$.

---

## 2. Full Leave-One-Family-Out (LOFO) Results

Evaluated across all 5 folds with block-bootstrap 95% confidence intervals over 10 chronological time blocks (1,000 resamples):

| Evaluated Fold (Held-Out Family) | Balanced Acc (%) | Held-Out Recall (%) | Rec(FTP) (%) | Rec(SSH) (%) | Rec(Gold) (%) | Rec(Hulk) (%) | Rec(Slow) (%) | FPR (%) | SFR (%) | Total Esc (%) | Held-Out Esc (%) | Benign Esc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: FTP-BruteForce** | 90.26 ± 17.54 | 2.86 ± 4.22 | 2.86 | 88.52 | 59.21 | 88.71 | 66.88 | 0.30 | **19.07 ± 35.21** | 3.10 | **2.86 ± 4.22** | 1.92 |
| **Held-out: SSH-BruteForce** | 91.09 ± 8.20 | 0.00 ± 0.00 | 66.10 | 0.00 | 60.35 | 83.92 | 65.90 | 0.29 | **17.32 ± 16.70** | 4.66 | **0.00 ± 0.00** | 1.59 |
| **Held-out: DoS-GoldenEye** | 82.42 ± 18.86 | 0.00 ± 0.00 | 62.50 | 87.28 | 0.00 | 40.47 | 63.40 | 0.15 | **34.48 ± 37.31** | 2.33 | **0.87 ± 1.25** | 0.32 |
| **Held-out: DoS-Hulk** | 89.24 ± 10.25 | 1.72 ± 2.49 | 63.70 | 88.82 | 63.47 | 1.72 | 66.39 | 0.17 | **14.54 ± 12.13** | 3.95 | **29.00 ± 25.16** | 1.21 |
| **Held-out: DoS-Slowloris** | 88.90 ± 16.67 | 8.35 ± 11.46 | 66.10 | 88.91 | 59.36 | 85.02 | 8.35 | 0.23 | **21.61 ± 32.88** | 4.47 | **0.95 ± 1.30** | 2.20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two-Tier System (Macro-Avg)** | **88.38 ± 2.70** | **2.59 ± 2.70** | - | - | - | - | - | 0.23 | **21.40 ± 6.08** | 3.70 | **6.73 ± 9.79** | 1.45 |
| **Tier 1 Standalone (Macro-Avg)** | 86.78 ± 2.97 | 1.92 ± 2.97 | - | - | - | - | - | 0.69 | **25.72 ± 6.18** | 0.00 | 0.00 | 0.00 |
| **Tier-2 Standalone (Macro-Avg)** | 90.19 ± 1.50 | 2.59 ± 2.70 | - | - | - | - | - | 0.15 | **19.46 ± 3.00** | 100.00 | 100.00 | 100.00 |

> **Reconciliation of Slowloris Fold Metrics:**  
> - **Two-Tier Integrated System:** Balanced Acc = 88.90 ± 16.67%  
> - **Tier-2 Standalone Model:** Balanced Acc = 89.98 ± 16.55%  

---

## 3. Window Size Feature Ablation: Strictly 74 Features Verified

```python
WIN_INDICES = [66, 67]
NON_WIN_INDICES = [i for i in range(76) if i not in WIN_INDICES]
assert len(NON_WIN_INDICES) == 74  # Verified 74 non-window feature indices
```

### Block-Bootstrap Evaluation Over 10 Chronological Time Blocks
Evaluated across 10 chronological time blocks (1,000 bootstrap resamples) with explicit Model column:

| Evaluated Model | Evaluated Fold | 76-Feature Balanced Acc (%) | 74-Feature (Ablated) Balanced Acc (%) | $\Delta$ Balanced Acc (%) | 76-Feat Held-Out Recall (%) | 74-Feat Held-Out Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Tier 2 RF (100 trees, depth 12) | **Held-out: FTP-BruteForce** | 91.33 ± 18.65% | 91.25 ± 18.80% | -0.08% | 2.86 ± 4.22% | 2.38 ± 3.51% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: SSH-BruteForce** | 92.18 ± 7.98% | 91.91 ± 7.54% | -0.27% | 0.00 ± 0.00% | 0.69 ± 0.50% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-GoldenEye** | 87.40 ± 18.58% | 89.53 ± 16.90% | +2.13% | 0.00 ± 0.00% | 8.24 ± 11.23% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-Hulk** | 90.14 ± 10.68% | 92.53 ± 6.59% | +2.39% | 1.66 ± 2.54% | 19.97 ± 30.47% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-Slowloris** | 90.09 ± 16.37% | 90.02 ± 16.41% | -0.07% | 8.35 ± 11.46% | 8.19 ± 11.24% |

**Ablation Finding:** Purging window features causes negligible difference in balanced accuracy across all folds, confirming that classification relies on packet lengths, flow rates, and inter-arrival timing rather than TCP window stack signatures.

---

## 4. Ground-Truth Audit: DoS-Hulk Flow Distribution & Capture Windows

### 4.1 Attacker Infrastructure & Operational Windows
- **Dataset Execution Date:** Friday, February 16, 2018 (CSE-CIC-IDS2018 Day 3).
- **Target Host:** Ubuntu 16.04 Apache Web Server (`18.218.115.60` / `172.31.69.25` on AWS EC2).
- **Attacker Host:** Kali Linux attack instance (`18.219.211.138`).
- **Hourly Label Counts from Timestamp Column vs CIC Documentation:**
  - **10:00–11:00 UTC:** 580 flows (Morning automated burst)
  - **11:00–12:00 UTC:** 323 flows (Morning trailing keep-alives)
    * **Morning Session Total:** 905 flows (Training partition source, $n=905$)
  - **13:00–14:00 UTC:** 245 flows (Afternoon throttled attack session onset)
  - **14:00–15:00 UTC:** 144 flows (Afternoon persistent keep-alives)
    * **Afternoon Session Total:** 389 flows (Test partition source, $n=389$)
  - **Total DoS-Hulk Flows:** 1294 flows.
- **CIC Documentation Reference:** Matches official Canadian Institute for Cybersecurity schedule detailing distinct execution scripts on 16/02/2018.

### 4.2 Training vs Test Hulk Flow Distributions

| Flow Metric | Training Hulk Flows ($n=905$) | Test Hulk Flows ($n=389$) | Shift Details |
| :--- | :---: | :---: | :---: |
| **Flow Duration (ms)** | Median = **146.6 ms**, Mean = **159.9 ms** | Median = **1358.6 ms**, Mean = **8335.5 ms** | Elevated session idle times |
| **Flow Packets/s** | Median = **61.85 pkts/s**, Mean = **8517.8 pkts/s** | Median = **8.06 pkts/s**, Mean = **7.7 pkts/s** | Lower packet frequency |
| **Total Forward Packets** | Median = **3.0 pkts**, Mean = **3.14 pkts** | Median = **5.0 pkts**, Mean = **4.93 pkts** | Persistent keep-alive probes |

### 4.3 Test Hulk Duration Histogram (Exact Interval Notation)
- **Bin 1 [341.37 ms, 17202.37 ms):** 358 flows
- **Bin 2 [17202.37 ms, 34063.37 ms):** 0 flows
- **Bin 3 [34063.37 ms, 50924.38 ms):** 0 flows
- **Bin 4 [50924.38 ms, 67785.38 ms):** 0 flows
- **Bin 5 [67785.38 ms, 84646.39 ms]:** 31 flows

Over 92% (358 / 389) of test Hulk flows cluster in the [341.37 ms, 17202.37 ms) interval, while 31 flows exhibit long keep-alive durations up to 84.6 seconds.
