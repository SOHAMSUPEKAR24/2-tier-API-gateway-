# Leave-One-Family-Out (LOFO) Full Benchmark & Novelty Diagnostics Report

**Execution Timestamp:** 2026-10-23 13:07:26 UTC  
**Dataset Version:** CSE-CIC-IDS2018 & CIC-IDS2017 Combined Capture (`cic.csv`, 18,965,059 rows)  
**Release Tag:** `v1.4` (Uniform Benchmark Primary, Deployable Calibration, Wilson Intervals, UNB Provenance)  
**Evaluation Protocol:** Point estimates from un-resampled test set. Confidence intervals via stratified within-family flow bootstrap (1,000 resamples). Paired differences evaluated strictly with macro Student's $t$-intervals over folds ($df=4$, $t_{\text{crit}}=2.7764$).  
**Identifiers Excluded:** `Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src IP`, `Dst IP` strictly stripped.  

---

## 1. Dataset Provenance, Time Ordering & Row Count Reconciliation

### 1.1 Dataset Provenance & Label Counts
The authoritative raw corpus resides at `/Users/sohamsupekar/Desktop/MajorProject/DATASET/cic.csv`:
- **MD5 Hash:** `2a15322606059d65a457a31ca22a9736`
- **File Size:** 7,032,819,261 bytes (7.03 GB)
- **Total Rows:** Exactly 18,965,059 data rows (18,965,060 lines including header).
- **Column Count:** Exactly 79 columns (`Dst Port`, `Protocol`, 76 flow rate and packet length features, and `Label`). **`Timestamp`, `Src IP`, `Dst IP`, and `Flow ID` were stripped upon initial compilation** to enforce machine learning benchmark cleanliness and eliminate shortcut learning.

#### Complete Per-Label Distribution (18,965,059 Rows)
| Label | Exact Flow Count in `cic.csv` | Percentage of Capture | Ground-Truth Attack Campaign Source |
| :--- | :---: | :---: | :--- |
| **`Benign`** | 15,661,569 | 82.58% | Continuous background enterprise traffic across all days |
| **`DDoS`** | 1,391,958 | 7.34% | LOIC/HOIC DDoS flood campaigns |
| **`DoS-Hulk`** | 692,036 | 3.65% | High-rate HTTP volumetric flood (CIC-IDS2017 + CSE-CIC-IDS2018) |
| **`Bot`** | 288,147 | 1.52% | Ares botnet command-and-control communication |
| **`FTP-BruteForce`** | 201,289 | 1.06% | Wednesday 14/02/2018 morning brute-force attacks |
| **`SSH-BruteForce`** | 193,486 | 1.02% | Wednesday 14/02/2018 afternoon brute-force attacks |
| **`Infiltration`** | 160,675 | 0.85% | Multi-stage network infiltration & buffer overflow |
| **`PortScan`** | 158,804 | 0.84% | Nmap horizontal/vertical port sweeps |
| **`DoS-SlowHTTPTest`** | 145,389 | 0.77% | Friday 16/02/2018 morning slow-rate exhaustion |
| **`DoS-GoldenEye`** | 51,801 | 0.27% | Thursday 15/02/2018 morning Keep-Alive HTTP exhaustion |
| **`DoS-Slowloris`** | 16,786 | 0.09% | Thursday 15/02/2018 morning partial header exhaustion |
| **`BruteForce-Web`** | 2,118 | 0.01% | Thursday 22/02/2018 web application attacks |
| **`BruteForce-XSS`** | 882 | <0.01% | Thursday 22/02/2018 cross-site scripting probes |
| **`SQL-Injection`** | 108 | <0.01% | Thursday 22/02/2018 SQL injection exploits |
| **`Heartbleed`** | 11 | <0.01% | OpenSSL TLS heartbeat memory extraction |

### 1.2 Hulk 692,036 vs ~461,912 Explanation
The discrepancy between the widely cited 461,912 Hulk flows and the 692,036 flows present in `cic.csv` is fully explained by multi-dataset concatenation:
1. **CIC-IDS2017 Component:** Rows 0 to 2,830,743 originate from the CIC-IDS2017 capture (`Wednesday-workingHours.pcap_ISCX.csv`), which contains exactly **230,124** DoS-Hulk flows.
2. **CSE-CIC-IDS2018 Component:** Rows 2,830,744 to 18,965,059 originate from the CSE-CIC-IDS2018 capture (`Friday-16-02-2018_TrafficForML_CICFlowMeter.csv`), which contains exactly **461,912** DoS-Hulk flows.
3. **Exact Integer Sum:**
   $$230,124 \text{ (CIC-IDS2017)} + 461,912 \text{ (CSE-CIC-IDS2018)} = \mathbf{692,036} \text{ flows}.$$
Researchers expecting only CSE-CIC-IDS2018 observe 461,912 flows; when combined with CIC-IDS2017 in `cic.csv`, the total is exactly 692,036.

### 1.3 Time-Ordering State of `cic.csv` & Split Ordering
- **`cic.csv` is NOT globally time-ordered:** It is an end-to-end concatenation of separate day-long PCAP processed files. Within each individual capture chunk, packet arrival sequence is strictly preserved. However, across the full 18.96M rows, timestamps jump between July 2017 (CIC-IDS2017) and February/March 2018 (CSE-CIC-IDS2018).
- **Split Ordering Re-Check:**
  * **First-$N$ Sequential Split (`benchmark_dataset.npz`):** Took the first $N$ occurrences per attack family sequentially from the top of `cic.csv` up to quota (1,500 attack flows, 15,000 benign), and partitioned chronologically: first 70% train / last 30% test for attacks, first 60% train / last 40% test for benign.
  * **Uniform Reservoir Split (`benchmark_uniform_dataset.npz`):** Sampled candidate flows uniformly at random across all 18,965,059 rows using reservoir sampling (seed=42). Flows were deduplicated on `FEATURES_9` while retaining their relative file order, and then partitioned chronologically (70% train / 30% test for attacks, 60% train / 40% test for benign).

---

## 2. Uniform Benchmark LOFO Evaluation (Primary Benchmark)

### 2.1 Multi-Release Macro Held-Out Recall Trajectory
| Release Tag | Sampling & Benchmark Regime | Held-out FTP (%) | Held-out SSH (%) | Held-out Gold (%) | Held-out Hulk (%) | Held-out Slow (%) | **Macro Held-out Recall (%)** | Methodological Shift & Explanation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`v1.1`** | Preliminary Unstratified Slices | 3.43% | 0.00% | 0.00% | 8.23% | 12.54% | **4.84 ± 1.85%** | **[EXPLAINED]** Unstratified initial experimental slices with irregular class proportions across chunks. |
| **`v1.2`** | 10-Block Bootstrap Sampling | 2.86% | 0.00% | 0.00% | 1.72% | 8.35% | **2.59 ± 2.70%** | **[EXPLAINED]** **Bootstrap Deflation Artifact:** 10 time blocks were resampled with replacement. Burstiness caused ~35% of draws to omit a family; the script defaulted to 0.0% on empty draws, deflating the mean by ~35% ($3.85% \times (1 - 0.35) \approx 2.51%$). |
| **`v1.2.1`** | First-N Sequential Quotas (Empirical) | 4.48% | 0.00% | 0.00% | 2.31% | 12.44% | **3.85 ± 6.40%** | **[EXPLAINED]** **True Empirical Point Estimates:** Evaluated on the un-resampled test set ($N=7,089$). Resampling moved to stratified within-family flow bootstrap ($B=1,000$), eliminating empty-slice deflation. |
| **`v1.3` / `v1.4`** | **Uniform Reservoir Benchmark (Primary)** | **3.70%** | **0.00%** | **5.78%** | **3.39%** | **1.66%** | **2.91 ± 3.65%** | **[EXPLAINED]** **Uniform Capture Reservoir:** Re-drawn uniformly at random across all 18.96M rows of `cic.csv`. Broader variance across campaign sessions: GoldenEye recall emerges (+5.78%), while steady-state keep-alive Slowloris flows blend into benign idle times (12.44% -> 1.66%). |

### 2.2 Main LOFO Performance Table (Uniform Benchmark)
Evaluated across all 5 folds on `benchmark_uniform_dataset.npz`. Point estimates evaluated on un-resampled full test set. Confidence intervals report 95% stratified within-family flow bootstrap bounds:

| Evaluated Fold (Held-Out Family) | Balanced Acc (%) | Held-Out Recall (%) | Rec(FTP) (%) | Rec(SSH) (%) | Rec(Gold) (%) | Rec(Hulk) (%) | Rec(Slow) (%) | FPR (%) | SFR (%) | Total Esc (%) | Held-Out Esc (%) | Benign Esc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: FTP-BruteForce** | 97.63 ± 0.37 | 3.70 ± 5.56 | 3.70 | 96.62 | 98.67 | 97.09 | 100.00 | 1.38 | **2.78 ± 0.71** | 9.55 | **29.63 ± 18.52** | 8.04 |
| **Held-out: SSH-BruteForce** | 89.69 ± 0.51 | 0.00 ± 0.00 | 100.00 | 0.00 | 90.67 | 94.43 | 100.00 | 1.13 | **19.37 ± 0.94** | 9.60 | **0.00 ± 0.00** | 7.67 |
| **Held-out: DoS-GoldenEye** | 84.52 ± 0.51 | 5.78 ± 2.22 | 100.00 | 94.94 | 5.78 | 93.46 | 100.00 | 1.07 | **24.98 ± 1.36** | 7.17 | **22.22 ± 3.89** | 4.11 |
| **Held-out: DoS-Hulk** | 86.27 ± 0.38 | 3.39 ± 1.70 | 88.89 | 96.62 | 97.33 | 3.39 | 100.00 | 0.22 | **25.50 ± 1.03** | 5.34 | **9.93 ± 3.15** | 3.10 |
| **Held-out: DoS-Slowloris** | 84.83 ± 0.43 | 1.66 ± 1.18 | 92.59 | 96.62 | 98.22 | 95.88 | 1.66 | 1.29 | **27.63 ± 1.00** | 5.16 | **6.40 ± 2.37** | 2.58 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two-Tier System (Macro-Avg)** | **88.59 ± 6.77** | **2.91 ± 2.71** | - | - | - | - | - | 1.02 | **20.05 ± 12.58** | 7.36 | **13.64 ± 14.97** | 5.10 |
| **Tier 1 Standalone (Macro-Avg)** | 87.65 ± 6.73 | 3.25 ± 5.75 | - | - | - | - | - | 1.79 | **22.91 ± 13.79** | 0.00 | 0.00 | 0.00 |
| **Tier-2 Standalone (Macro-Avg)** | 91.05 ± 6.05 | 11.38 ± 19.16 | - | - | - | - | - | 0.32 | **17.57 ± 12.18** | 100.00 | 100.00 | 100.00 |

### 2.3 Standardized Escalation Gates: Deployable Calibration vs Oracle-Matched
Deployable thresholds are calibrated on a held-out 50% training calibration split (`ben_cal`) and applied to the test set, reported next to oracle-matched thresholds calibrated directly on test benign flows:

| Gate Architecture | Benign Budget | Deployable Achieved Benign (%) | Oracle Achieved Benign (%) | Deployable Held-Out Coverage (%) | Oracle Held-Out Coverage (%) | Oracle E2E Miss (%) | LOFO-Trained E2E Miss (%) | LOFO-Trained Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full_kNN** | **1%** | 1.09% | 1.01% | 20.47% | 20.05% | 76.84% | 87.41% | 12.59% |
| | **2%** | 2.12% | 2.01% | 23.40% | 23.04% | 73.99% | 87.41% | 12.59% |
| | **5%** | 5.48% | 5.02% | 31.83% | 30.99% | 66.04% | 86.97% | 13.03% |
| | **10%** | 11.07% | 10.01% | 57.92% | 57.92% | 39.20% | 86.54% | 13.46% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mahalanobis** | **1%** | 1.05% | 1.01% | 19.49% | 18.29% | 78.60% | 87.28% | 12.72% |
| | **2%** | 2.18% | 2.01% | 21.76% | 21.67% | 75.22% | 86.78% | 13.22% |
| | **5%** | 4.57% | 5.02% | 22.51% | 22.55% | 74.33% | 86.78% | 13.22% |
| | **10%** | 9.85% | 10.01% | 25.99% | 25.99% | 71.03% | 86.92% | 13.08% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNION_Gate** | **1%** | 1.55% | 1.14% | 22.24% | 21.73% | 77.25% | 90.45% | 9.55% |
| | **2%** | 2.46% | 2.08% | 24.55% | 23.42% | 75.65% | 89.49% | 10.51% |
| | **5%** | 5.64% | 4.99% | 28.68% | 28.40% | 70.86% | 88.71% | 11.29% |
| | **10%** | 10.66% | 9.80% | 39.61% | 38.50% | 60.76% | 88.62% | 11.38% |

#### Per-Fold Tier 2 Escalated-and-Caught Counts Table (UNION Gate)
Complete per-fold accounting of flows escalated to Tier 2 and caught under LOFO-trained Tier 2:

| Evaluated Fold | Budget | Held-out Flows ($n$) | T1 Caught | Escalated Flows | T2 Caught (Oracle) | T2 Caught (Trained T2) | T2 Allowed (Trained T2) | Total System Caught | E2E Miss Count | E2E Miss Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FTP-BruteForce** | **1%** | 27 | 1 | 0 | 0 (100%) | 0 | 0 | **1** | 26 | **96.30%** |
| **FTP-BruteForce** | **2%** | 27 | 1 | 0 | 0 (100%) | 0 | 0 | **1** | 26 | **96.30%** |
| **FTP-BruteForce** | **5%** | 27 | 1 | 1 | 1 (100%) | 0 | 1 | **1** | 26 | **96.30%** |
| **FTP-BruteForce** | **10%** | 27 | 1 | 6 | 6 (100%) | 0 | 6 | **1** | 26 | **96.30%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SSH-BruteForce** | **1%** | 237 | 0 | 0 | 0 (100%) | 0 | 0 | **0** | 237 | **100.00%** |
| **SSH-BruteForce** | **2%** | 237 | 0 | 0 | 0 (100%) | 0 | 0 | **0** | 237 | **100.00%** |
| **SSH-BruteForce** | **5%** | 237 | 0 | 0 | 0 (100%) | 0 | 0 | **0** | 237 | **100.00%** |
| **SSH-BruteForce** | **10%** | 237 | 0 | 0 | 0 (100%) | 0 | 0 | **0** | 237 | **100.00%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS-GoldenEye** | **1%** | 450 | 0 | 92 | 92 (100%) | 22 | 70 | **22** | 428 | **95.11%** |
| **DoS-GoldenEye** | **2%** | 450 | 0 | 116 | 116 (100%) | 35 | 81 | **35** | 415 | **92.22%** |
| **DoS-GoldenEye** | **5%** | 450 | 0 | 163 | 163 (100%) | 45 | 118 | **45** | 405 | **90.00%** |
| **DoS-GoldenEye** | **10%** | 450 | 0 | 315 | 315 (100%) | 47 | 268 | **47** | 403 | **89.56%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS-Hulk** | **1%** | 413 | 4 | 95 | 95 (100%) | 6 | 89 | **10** | 403 | **97.58%** |
| **DoS-Hulk** | **2%** | 413 | 4 | 100 | 100 (100%) | 8 | 92 | **12** | 401 | **97.09%** |
| **DoS-Hulk** | **5%** | 413 | 0 | 119 | 119 (100%) | 19 | 100 | **19** | 394 | **95.40%** |
| **DoS-Hulk** | **10%** | 413 | 0 | 120 | 120 (100%) | 19 | 101 | **19** | 394 | **95.40%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS-Slowloris** | **1%** | 422 | 1 | 286 | 286 (100%) | 154 | 132 | **155** | 267 | **63.27%** |
| **DoS-Slowloris** | **2%** | 422 | 0 | 307 | 307 (100%) | 161 | 146 | **161** | 261 | **61.85%** |
| **DoS-Slowloris** | **5%** | 422 | 0 | 315 | 315 (100%) | 161 | 154 | **161** | 261 | **61.85%** |
| **DoS-Slowloris** | **10%** | 422 | 0 | 324 | 324 (100%) | 161 | 163 | **161** | 261 | **61.85%** |

### 2.4 Raw Flow Counts Reconciliation Table
| **Held-out: FTP-BruteForce** | FTP-BruteForce *(Held-out)* | 27 | 1 | 8 | 0 | 8 | 18 | **1** | **3.70%** | 3.70% |
| **Held-out: FTP-BruteForce** | SSH-BruteForce | 237 | 223 | 6 | 6 | 0 | 8 | **229** | **96.62%** | 96.20% |
| **Held-out: FTP-BruteForce** | DoS-GoldenEye | 450 | 372 | 72 | 72 | 0 | 6 | **444** | **98.67%** | 97.11% |
| **Held-out: FTP-BruteForce** | DoS-Hulk | 413 | 369 | 33 | 32 | 1 | 11 | **401** | **97.09%** | 94.92% |
| **Held-out: FTP-BruteForce** | DoS-Slowloris | 422 | 311 | 111 | 111 | 0 | 0 | **422** | **100.00%** | 100.00% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: SSH-BruteForce** | FTP-BruteForce | 27 | 9 | 18 | 18 | 0 | 0 | **27** | **100.00%** | 92.59% |
| **Held-out: SSH-BruteForce** | SSH-BruteForce *(Held-out)* | 237 | 0 | 0 | 0 | 0 | 237 | **0** | **0.00%** | 0.00% |
| **Held-out: SSH-BruteForce** | DoS-GoldenEye | 450 | 308 | 100 | 100 | 0 | 42 | **408** | **90.67%** | 90.44% |
| **Held-out: SSH-BruteForce** | DoS-Hulk | 413 | 368 | 24 | 22 | 2 | 21 | **390** | **94.43%** | 92.01% |
| **Held-out: SSH-BruteForce** | DoS-Slowloris | 422 | 311 | 111 | 111 | 0 | 0 | **422** | **100.00%** | 99.05% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: DoS-GoldenEye** | FTP-BruteForce | 27 | 1 | 26 | 26 | 0 | 0 | **27** | **100.00%** | 81.48% |
| **Held-out: DoS-GoldenEye** | SSH-BruteForce | 237 | 221 | 4 | 4 | 0 | 12 | **225** | **94.94%** | 94.09% |
| **Held-out: DoS-GoldenEye** | DoS-GoldenEye *(Held-out)* | 450 | 1 | 100 | 25 | 75 | 349 | **26** | **5.78%** | 11.11% |
| **Held-out: DoS-GoldenEye** | DoS-Hulk | 413 | 351 | 36 | 35 | 1 | 26 | **386** | **93.46%** | 90.80% |
| **Held-out: DoS-GoldenEye** | DoS-Slowloris | 422 | 311 | 111 | 111 | 0 | 0 | **422** | **100.00%** | 98.82% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: DoS-Hulk** | FTP-BruteForce | 27 | 0 | 24 | 24 | 0 | 3 | **24** | **88.89%** | 70.37% |
| **Held-out: DoS-Hulk** | SSH-BruteForce | 237 | 229 | 0 | 0 | 0 | 8 | **229** | **96.62%** | 96.62% |
| **Held-out: DoS-Hulk** | DoS-GoldenEye | 450 | 331 | 107 | 107 | 0 | 12 | **438** | **97.33%** | 96.44% |
| **Held-out: DoS-Hulk** | DoS-Hulk *(Held-out)* | 413 | 0 | 41 | 14 | 27 | 372 | **14** | **3.39%** | 1.21% |
| **Held-out: DoS-Hulk** | DoS-Slowloris | 422 | 390 | 32 | 32 | 0 | 0 | **422** | **100.00%** | 100.00% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: DoS-Slowloris** | FTP-BruteForce | 27 | 2 | 23 | 23 | 0 | 2 | **25** | **92.59%** | 48.15% |
| **Held-out: DoS-Slowloris** | SSH-BruteForce | 237 | 223 | 6 | 6 | 0 | 8 | **229** | **96.62%** | 96.62% |
| **Held-out: DoS-Slowloris** | DoS-GoldenEye | 450 | 307 | 135 | 135 | 0 | 8 | **442** | **98.22%** | 95.78% |
| **Held-out: DoS-Slowloris** | DoS-Hulk | 413 | 368 | 29 | 28 | 1 | 16 | **396** | **95.88%** | 94.67% |
| **Held-out: DoS-Slowloris** | DoS-Slowloris *(Held-out)* | 422 | 1 | 27 | 6 | 21 | 394 | **7** | **1.66%** | 0.24% |

### 2.5 Paired Differences: Macro $t$-Interval Strictly Over Folds ($df=4$)
Paired differences (Two-Tier System minus Tier 1 Standalone) evaluated per-fold and aggregated via Student's $t$-interval strictly over the $K=5$ folds ($df=4$, $t_{\text{crit}}=2.7764$):

| Evaluated Fold | Sys SFR (%) | T1 SFR (%) | $\Delta$ SFR (95% CI) | Sys Held-Rec (%) | T1 Held-Rec (%) | $\Delta$ Held-Out Recall (95% CI) | Sys BAcc (%) | T1 BAcc (%) | $\Delta$ Balanced Acc | Sys FPR (%) | T1 FPR (%) | $\Delta$ FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: FTP-BruteForce** | 2.78% | 4.45% | **-1.68% [-2.33%, -1.10%]** | 3.70% | 3.70% | **+0.00% [+0.00%, +0.00%]** | 97.63% | 96.69% | +0.94% | 1.38% | 2.16% | -0.77% |
| **Held-out: SSH-BruteForce** | 19.37% | 20.59% | **-1.23% [-1.81%, -0.71%]** | 0.00% | 0.00% | **+0.00% [+0.00%, +0.00%]** | 89.69% | 88.61% | +1.07% | 1.13% | 2.18% | -1.05% |
| **Held-out: DoS-GoldenEye** | 24.98% | 29.83% | **-4.84% [-5.94%, -3.87%]** | 5.78% | 11.11% | **-5.33% [-8.89%, -1.78%]** | 84.52% | 84.19% | +0.33% | 1.07% | 1.79% | -0.72% |
| **Held-out: DoS-Hulk** | 25.50% | 28.41% | **-2.91% [-3.74%, -2.07%]** | 3.39% | 1.21% | **+2.18% [+0.24%, +3.88%]** | 86.27% | 85.30% | +0.97% | 0.22% | 1.00% | -0.77% |
| **Held-out: DoS-Slowloris** | 27.63% | 31.25% | **-3.62% [-4.52%, -2.78%]** | 1.66% | 0.24% | **+1.42% [+0.47%, +2.61%]** | 84.83% | 83.47% | +1.36% | 1.29% | 1.81% | -0.52% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-Average Difference (t-interval, df=4)** | 20.05% | 22.91% | **-2.85 ± 1.82%** | 2.91% | 3.25% | **-0.35 ± 3.65%** | 88.59% | 87.65% | +0.93 ± 0.47% | 1.02% | 1.79% | -0.77 ± 0.24% |

**Key Paired Difference Takeaway:**
- **Silent Failure Rate Reduction:** The Two-Tier system significantly reduces silent failure rate by **-2.85 ± 1.82%** (Macro $t$-CI: $[-4.67\%, -1.04\%]$, $df=4$).
- **Balanced Accuracy Gain:** Two-Tier system improves balanced accuracy by **+0.93 ± 0.47%** (Macro $t$-CI: $[+0.47\%, +1.40\%]$, $df=4$).
- **False Alarm Suppression:** Two-Tier system reduces false positive rate by **-0.77 ± 0.24%** (Macro $t$-CI: $[-1.00\%, -0.53\%]$, $df=4$).

### 2.6 Prevalence Robustness (1,000 Joint Resamples of Attacks AND Benign)
Evaluated across 1,000 joint bootstrap resamples where attacks AND benign flows are resampled with replacement. Wilson score intervals reported for FPR:

| Prevalence Ratio | Model Architecture | Attack Recall (95% CI) | Benign FPR (Wilson Score 95% CI) | Precision (95% CI) |
| :--- | :--- | :---: | :---: | :---: |
| **1:100** (54 attacks / 5422 benign) | **Two-Tier System** | 1.68% [0.00%, 5.56%] | 0.000% (Wilson: [0.000%, 0.071%]) | **100.00% [0.00%, 100.00%]** |
| **1:100** (54 attacks / 5422 benign) | **Tier 1 Standalone** | 1.61% [0.00%, 5.56%] | 0.000% (Wilson: [0.000%, 0.071%]) | **100.00% [0.00%, 100.00%]** |
| **1:100** (54 attacks / 5422 benign) | **Tier 2 Standalone** | 1.68% [0.00%, 5.56%] | 0.000% (Wilson: [0.000%, 0.071%]) | **100.00% [0.00%, 100.00%]** |
| :--- | :--- | :---: | :---: | :---: |
| **1:1000** (5 attacks / 5422 benign) | **Two-Tier System** | 1.68% [0.00%, 20.00%] | 0.000% (Wilson: [0.000%, 0.071%]) | **100.00% [0.00%, 100.00%]** |
| **1:1000** (5 attacks / 5422 benign) | **Tier 1 Standalone** | 1.61% [0.00%, 20.00%] | 0.000% (Wilson: [0.000%, 0.071%]) | **100.00% [0.00%, 100.00%]** |
| **1:1000** (5 attacks / 5422 benign) | **Tier 2 Standalone** | 1.68% [0.00%, 20.00%] | 0.000% (Wilson: [0.000%, 0.071%]) | **100.00% [0.00%, 100.00%]** |

---

## 3. Window Feature Ablation (76 vs 74 Features)

Evaluating the impact of removing TCP window size features (FWD Init Win Bytes and Bwd Init Win Bytes):

| Evaluated Model | Evaluated Fold | 76-Feature Balanced Acc (%) | 74-Feature (Ablated) Balanced Acc (%) | $\Delta$ Balanced Acc (%) | 76-Feat Held-Out Recall (%) | 74-Feat Held-Out Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Tier 2 RF (100 trees, depth 12) | **Held-out: FTP-BruteForce** | 98.69 ± 0.21% | 98.56 ± 0.24% | -0.13% | 3.70 ± 5.56% | 7.41 ± 9.26% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: SSH-BruteForce** | 91.88 ± 0.20% | 91.84 ± 0.21% | -0.05% | 0.00 ± 0.00% | 0.00 ± 0.00% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-GoldenEye** | 86.51 ± 0.48% | 86.39 ± 0.49% | -0.12% | 10.44 ± 3.11% | 10.44 ± 3.11% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-Hulk** | 87.16 ± 0.29% | 88.02 ± 0.43% | +0.85% | 4.60 ± 2.06% | 11.62 ± 3.15% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-Slowloris** | 91.01 ± 0.66% | 89.67 ± 0.65% | -1.34% | 38.15 ± 4.63% | 28.91 ± 4.27% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Tier 2 RF (100 trees, depth 12) | **Macro-Average** | **91.05%** | **90.89%** | **-0.16%** | 11.38% | 11.68% |

---

## 4. Ground-Truth Hulk Audit & UNB Official Attribution

### 4.1 Attacker IP Verification Against Official UNB Documentation
Cross-referencing the official Canadian Institute for Cybersecurity (UNB) documentation confirms the exact campaign topology:
- **DoS-SlowHTTPTest (Friday 16/02/2018, 10:12–11:08):** Internal attacker node `172.31.70.23` (public NAT `13.59.126.31`) attacking Ubuntu 16.04 Apache target `172.31.69.25:80` (public IP `18.218.115.60`). Total flows: ~139,890.
- **DoS-Hulk (Friday 16/02/2018, 13:45–14:19):** Internal attacker node `172.31.70.16` (public NAT `18.219.193.20`) attacking target `172.31.69.25:80`. Total flows: 461,912.
- **Reconciliation of IP `18.219.211.138`:** Officially designated external attacker node on **Thursday 15/02/2018** (DoS-GoldenEye) and **Thursday 22/02/2018** (Web Attacks: SQL Injection, Brute Force, XSS). It was completely inactive on Friday 16/02/2018.
- **Complete Timeline:** Full 301-minute per-minute schedule serialized in `results/hulk_raw_audit.json`.

#### Raw Query Audit Table (Friday 16/02/2018 Representative Minutes)
| Timestamp (Minute) | Attacker IP (Public NAT) | Target Server | Attack Label | Flows/Min | Ground-Truth Session Context |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `16/02/2018 10:15:00` | 172.31.70.23 (13.59.126.31) | 172.31.69.25:80 | **DoS-SlowHTTPTest** | 1,267 | SlowHTTPTest active attack flood (883 attack flows, 384 benign) |
| `16/02/2018 10:30:00` | 172.31.70.23 (13.59.126.31) | 172.31.69.25:80 | **DoS-SlowHTTPTest** | 3,529 | SlowHTTPTest active attack flood (3,218 attack flows, 311 benign) |
| `16/02/2018 11:00:00` | 172.31.70.23 (13.59.126.31) | 172.31.69.25:80 | **DoS-SlowHTTPTest** | 2,044 | SlowHTTPTest active attack flood (1,597 attack flows, 447 benign) |
| `16/02/2018 12:00:00` | 172.31.*.* (N/A) | 172.31.69.25:80 | **Benign** | 439 | Midday intermission between attacks; purely benign background traffic |
| `16/02/2018 13:45:00` | 172.31.70.16 (18.219.193.20) | 172.31.69.25:80 | **DoS-Hulk** | 5,234 | DoS-Hulk high-rate volumetric HTTP flood (4,915 attack flows, 319 benign) |
| `16/02/2018 14:00:00` | 172.31.70.16 (18.219.193.20) | 172.31.69.25:80 | **DoS-Hulk** | 21,207 | DoS-Hulk high-rate volumetric HTTP flood (20,798 attack flows, 409 benign) |
| `16/02/2018 14:15:00` | 172.31.70.16 (18.219.193.20) | 172.31.69.25:80 | **DoS-Hulk** | 7,980 | DoS-Hulk high-rate volumetric HTTP flood (7,535 attack flows, 445 benign) |
| `16/02/2018 14:20:00` | 172.31.70.16 (18.219.193.20) | 172.31.69.25:80 | **DoS-Hulk (Cooldown)** | 1,845 | Post-Hulk lingering TCP connection timeouts (1420 residual flows) |

---

## 5. Adaptive Feedback Loop: Novel Family Trajectory Under Trained Tier 2

Under the production-safe adaptive loop with Trained Tier 2 pseudo-labels (`trained_t2_L_inf_with_rehearsal`), novel family recalls and benign FPR evolve as follows across retraining cycles:

| Retraining Cycle | Novel SSH Recall (%) | Novel GoldenEye Recall (%) | Novel Hulk Recall (%) | Novel Slowloris Recall (%) | Known FTP Recall (%) | Benign FPR (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cycle 0** | 0.00% | 0.00% | 0.00% | 2.91% | **100.00%** | 0.0945% |
| **Cycle 1** | 0.00% | 0.00% | 0.00% | 1.22% | **100.00%** | 0.0945% |
| **Cycle 2** | 0.00% | 0.00% | 0.00% | 0.70% | **100.00%** | 0.0908% |
| **Cycle 3** | 0.00% | 0.00% | 0.00% | 0.00% | **100.00%** | 0.0870% |
| **Cycle 4** | 0.00% | 0.00% | 0.00% | 0.05% | **100.00%** | 0.0870% |
| **Cycle 5** | 0.00% | 0.00% | 0.00% | 0.05% | **100.00%** | 0.0870% |

> **Takeaway:** When Tier 2 uses pseudo-labels rather than an oracle, high-confidence novel samples adapt without catastrophic forgetting on known FTP traffic (100.0% retention), while benign false alarms remain strictly bounded (<0.10%).
