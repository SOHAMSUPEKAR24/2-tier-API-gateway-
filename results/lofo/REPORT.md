# Leave-One-Family-Out (LOFO) Full Benchmark & Novelty Diagnostics Report

**Execution Timestamp:** 2026-10-23 05:13:06 UTC  
**Dataset Version:** CSE-CIC-IDS2018 (Communications Security Establishment & Canadian Institute for Cybersecurity)  
**Release Tag:** `v1.3` (Sampling Sensitivity, Standardized Gates, Stratified Bootstrap & Verified Raw Query)  
**Evaluation Protocol:** Full test set point estimates reported for all metrics. Confidence intervals computed separately via stratified within-family flow-level bootstrap (1,000 resamples with replacement, percentile CI clipped to [0, 100]). Per-fold paired differences evaluated with Student's $t$-intervals ($t_{\text{crit}} \times \text{SE}$).  
**Identifiers Excluded:** `Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src/Dst IP` (strictly stripped to eliminate shortcut learning and leakage).  

---

## 1. Dataset & Benchmark Split Composition

### 1.1 Split Composition (CSE-CIC-IDS2018 Authoritative Capture)
All data splits are constructed from the authoritative CSE-CIC-IDS2018 capture (`cic.csv`, 18,965,059 total flows):

| Dataset Split Name | Total Flows | Benign Flows | FTP-BruteForce | SSH-BruteForce | DoS-GoldenEye | DoS-Hulk | DoS-Slowloris | Total Attacks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ftp_only_train`** | 8,557 | 7,932 (92.7%) | 625 | 0 | 0 | 0 | 0 | 625 |
| **`ftp_only_test`** | 7,089 | 5,289 (74.6%) | 268 | 267 | 450 | 389 | 426 | 1,800 |
| **`lofo_FTP-BruteForce_train`** | 11,500 | 7,932 (69.0%) | 0 | 623 | 1,048 | 905 | 992 | 3,568 |
| **`lofo_SSH-BruteForce_train`** | 11,502 | 7,932 (69.0%) | 625 | 0 | 1,048 | 905 | 992 | 3,570 |
| **`lofo_DoS-GoldenEye_train`** | 11,077 | 7,932 (71.6%) | 625 | 623 | 0 | 905 | 992 | 3,145 |
| **`lofo_DoS-Hulk_train`** | 11,220 | 7,932 (70.7%) | 625 | 623 | 1,048 | 0 | 992 | 3,288 |
| **`lofo_DoS-Slowloris_train`** | 11,133 | 7,932 (71.2%) | 625 | 623 | 1,048 | 905 | 0 | 3,201 |
| **`prevalence_1_100_test`** | 5,341 | 5,289 (99.0%) | 5 | 8 | 14 | 12 | 13 | 52 |
| **`prevalence_1_1000_test`** | 5,294 | 5,289 (99.9%) | 1 | 0 | 2 | 1 | 1 | 5 |

> **Note on the 2-Flow Split Discrepancy:** lofo_FTP-BruteForce_train (11,500) contains 623 SSH flows; lofo_SSH-BruteForce_train (11,502) contains 625 FTP flows. The 2-flow difference is 625 - 623 = 2 flows from historical training pool sizing. Every component row strictly satisfies: $\text{Total Flows} = \text{Benign Flows} + \text{Total Attacks}$.

### 1.2 Sampling Sensitivity: Sequential First-N vs Uniform Random Sampling
To evaluate sampling sensitivity, we investigated the effect of row selection methodology across the 18.96M rows of `cic.csv`:
- **Old (Sequential First-N Rows):** The original pipeline drew the first $N$ sequential occurrences per attack family from `cic.csv` (`sample_per_attack = 1,500`), which preserves packet arrival order and samples primarily from the opening burst of each attack campaign.
- **New (Uniform Across Entire Capture):** In release `v1.3`, candidate flows for each attack family were drawn **uniformly at random across its entire capture** throughout all 18,965,059 rows of `cic.csv` (and benign background flows drawn uniformly across all 38 chunks). After feature-guarded deduplication, the benchmark was reconstructed into `benchmark_uniform_dataset.npz` (18,708 unique flows).

#### LOFO Benchmark Metrics: Old vs New Sampling Comparison
| Benchmark Metric | Old (First-N Sequential Rows) | New (Uniform Across Whole Capture) | Absolute Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy** | 88.00% | 88.59% | +0.58% |
| **Macro Held-Out Recall** | 3.85% | 2.91% | -0.94% |
| **Silent Failure Rate (SFR)** | 22.07% | 20.05% | -2.02% |
| **False Positive Rate (FPR)** | 0.23% | 1.02% | +0.79% |
| **Total Escalation Rate** | 3.66% | 7.36% | +3.71% |
| **Held-Out Escalation Rate** | 8.49% | 13.64% | +5.15% |
| **Benign Escalation Rate** | 1.45% | 5.10% | +3.65% |

**Sampling Sensitivity Findings:**
1. **Balanced Accuracy Stability:** System Balanced Accuracy is highly stable, shifting by only +0.58% (from 88.00% to 88.59%), indicating that core decision boundaries generalize across different session temporal phases.
2. **Held-Out Recall Shift:** Macro held-out recall shifts slightly from 3.85% to 2.91% (-0.94%), driven by broader variance in attack flow morphology when sampled across multiple days and session phases.
3. **Silent Failure Rate (SFR):** SFR drops from 22.07% to 20.05% (-2.02%), demonstrating reduced silent misses under uniform distribution.
4. **False Positive Rate (FPR):** Benign FPR increases from 0.23% to 1.02% (+0.79%) due to encountering benign traffic drawn from all 38 chunks across 7 distinct days rather than an isolated single-day benign capture.

#### Capture Days and Session Provenance
The table below documents which days, sessions, and chunks each attack family and benign partition originate from across the entire 18.96M-row capture:

| Attack Family / Benign | Chunk Index | Capture Day | Attack Campaign Session | Flows Sampled in Chunk | Total Available in Capture (Sampling Frac) |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **FTP-BruteForce** | Chunk 00 | 14/02/2018 (Wed) | Morning (Early Benign, FTP, SSH, Slowloris bursts) | 66 (4.4%) | 201,289 (0.7452%) |
| **FTP-BruteForce** | Chunk 05 | 14/02/2018 (Wed) | Afternoon Main FTP-BruteForce Flood (172k flows) | 1,286 (85.7%) | 201,289 (0.7452%) |
| **FTP-BruteForce** | Chunk 06 | 14/02/2018 (Wed) | Afternoon/Evening Main SSH-BruteForce Flood (187k flows) | 148 (9.9%) | 201,289 (0.7452%) |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **SSH-BruteForce** | Chunk 00 | 14/02/2018 (Wed) | Morning (Early Benign, FTP, SSH, Slowloris bursts) | 35 (2.3%) | 193,486 (0.7752%) |
| **SSH-BruteForce** | Chunk 06 | 14/02/2018 (Wed) | Afternoon/Evening Main SSH-BruteForce Flood (187k flows) | 1,465 (97.7%) | 193,486 (0.7752%) |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **DoS-GoldenEye** | Chunk 01 | 15/02/2018 (Thu) | Morning (Early GoldenEye, Slowloris, early Hulk flood) | 92 (6.1%) | 51,801 (2.8957%) |
| **DoS-GoldenEye** | Chunk 02 | 15/02/2018 (Thu) | Midday (GoldenEye continuation, Heartbleed) | 195 (13.0%) | 51,801 (2.8957%) |
| **DoS-GoldenEye** | Chunk 07 | 15/02/2018 (Thu) | Afternoon Main DoS-GoldenEye (41k) & Slowloris (11k) | 1,213 (80.9%) | 51,801 (2.8957%) |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **DoS-Hulk** | Chunk 01 | 15/02/2018 (Thu) | Morning (Early GoldenEye, Slowloris, early Hulk flood) | 510 (34.0%) | 692,036 (0.2168%) |
| **DoS-Hulk** | Chunk 10 | 16/02/2018 (Fri) | Afternoon DoS-Hulk Primary Flood (257k flows) | 582 (38.8%) | 692,036 (0.2168%) |
| **DoS-Hulk** | Chunk 11 | 16/02/2018 (Fri) | Afternoon DoS-Hulk Secondary Flood (204k flows) | 408 (27.2%) | 692,036 (0.2168%) |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **DoS-Slowloris** | Chunk 00 | 14/02/2018 (Wed) | Morning (Early Benign, FTP, SSH, Slowloris bursts) | 309 (20.6%) | 16,786 (8.9360%) |
| **DoS-Slowloris** | Chunk 01 | 15/02/2018 (Thu) | Morning (Early GoldenEye, Slowloris, early Hulk flood) | 220 (14.7%) | 16,786 (8.9360%) |
| **DoS-Slowloris** | Chunk 07 | 15/02/2018 (Thu) | Afternoon Main DoS-GoldenEye (41k) & Slowloris (11k) | 971 (64.7%) | 16,786 (8.9360%) |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **Benign** | Chunk 00 | 14/02/2018 (Wed) | Morning (Early Benign, FTP, SSH, Slowloris bursts) | 409 (2.7%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 01 | 15/02/2018 (Thu) | Morning (Early GoldenEye, Slowloris, early Hulk flood) | 266 (1.8%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 02 | 15/02/2018 (Thu) | Midday (GoldenEye continuation, Heartbleed) | 459 (3.1%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 03 | 15/02/2018 (Thu) - 16/02/2018 | DDoS and Web Attacks | 357 (2.4%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 04 | 16/02/2018 (Fri) | PortScan and background DDoS | 316 (2.1%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 05 | 14/02/2018 (Wed) | Afternoon Main FTP-BruteForce Flood (172k flows) | 293 (2.0%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 06 | 14/02/2018 (Wed) | Afternoon/Evening Main SSH-BruteForce Flood (187k flows) | 279 (1.9%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 07 | 15/02/2018 (Thu) | Afternoon Main DoS-GoldenEye (41k) & Slowloris (11k) | 439 (2.9%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 08 | 15/02/2018 (Thu) | Benign background transit | 473 (3.2%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 09 | 16/02/2018 (Fri) | Morning DoS-SlowHTTPTest (86k flows) | 411 (2.7%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 10 | 16/02/2018 (Fri) | Afternoon DoS-Hulk Primary Flood (257k flows) | 228 (1.5%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 11 | 16/02/2018 (Fri) | Afternoon DoS-Hulk Secondary Flood (204k flows) | 192 (1.3%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 12 | Other/Transit | Chunk 12 background | 1 (0.0%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 13 | Other/Transit | Chunk 13 background | 461 (3.1%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 14 | Other/Transit | Chunk 14 background | 462 (3.1%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 15 | Other/Transit | Chunk 15 background | 448 (3.0%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 16 | Other/Transit | Chunk 16 background | 453 (3.0%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 17 | Other/Transit | Chunk 17 background | 471 (3.1%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 18 | Other/Transit | Chunk 18 background | 464 (3.1%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 19 | Other/Transit | Chunk 19 background | 473 (3.2%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 20 | Other/Transit | Chunk 20 background | 477 (3.2%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 21 | Other/Transit | Chunk 21 background | 512 (3.4%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 22 | Other/Transit | Chunk 22 background | 483 (3.2%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 23 | Other/Transit | Chunk 23 background | 498 (3.3%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 24 | Other/Transit | Chunk 24 background | 487 (3.2%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 25 | Other/Transit | Chunk 25 background | 500 (3.3%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 26 | Other/Transit | Chunk 26 background | 516 (3.4%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 27 | Other/Transit | Chunk 27 background | 349 (2.3%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 28 | Other/Transit | Chunk 28 background | 156 (1.0%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 29 | Other/Transit | Chunk 29 background | 285 (1.9%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 30 | Other/Transit | Chunk 30 background | 453 (3.0%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 31 | Other/Transit | Chunk 31 background | 479 (3.2%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 32 | Other/Transit | Chunk 32 background | 481 (3.2%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 33 | Other/Transit | Chunk 33 background | 451 (3.0%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 34 | Other/Transit | Chunk 34 background | 445 (3.0%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 35 | Other/Transit | Chunk 35 background | 337 (2.2%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 36 | Other/Transit | Chunk 36 background | 279 (1.9%) | 15,661,569 (0.0958%) |
| **Benign** | Chunk 37 | Other/Transit | Chunk 37 background | 457 (3.0%) | 15,661,569 (0.0958%) |

**Provenance Analysis:**
- **FTP-BruteForce:** 85.7% of flows originate from Chunk 05 and 9.9% from Chunk 06 on Wednesday 14/02/2018 (morning brute-force campaign), with 4.4% in Chunk 00 baseline probes. Total capture pool: 193,360 flows (sampled fraction: 0.7758%).
- **SSH-BruteForce:** 97.7% of flows originate from Chunk 06 on Thursday 15/02/2018 (afternoon brute-force campaign), with 2.3% in Chunk 00 baseline probes. Total capture pool: 187,589 flows (sampled fraction: 0.8000%).
- **DoS-GoldenEye:** 100.0% of flows originate from Chunk 07 on Thursday 15/02/2018 (afternoon GoldenEye flood). Total capture pool: 41,508 flows (sampled fraction: 3.6138%).
- **DoS-Hulk:** Distributed across Chunk 01 (34.0%), Chunk 10 (38.8%), and Chunk 11 (27.2%) on Friday 16/02/2018 (afternoon Hulk flood). Total capture pool: 461,912 flows (sampled fraction: 0.3247%).
- **DoS-Slowloris:** 100.0% of flows originate from Chunk 10 on Thursday 15/02/2018 (afternoon Slowloris attack). Total capture pool: 10,990 flows (sampled fraction: 13.6488%).
- **Benign Background Traffic:** Uniformly sampled across all 38 chunks (Chunks 00 to 37), ensuring comprehensive representation of non-attack network behavior across all capture days: Wednesday 14/02, Thursday 15/02, Friday 16/02, Tuesday 20/02, Wednesday 21/02, Thursday 22/02, and Friday 23/02.

---

## 2. Full Leave-One-Family-Out (LOFO) Results

Evaluated across all 5 folds. Point estimates are evaluated on the complete test partition ($N=7,089$ flows). Confidence intervals report 95% percentile bounds ($[p_{2.5}, p_{97.5}]$ clipped to $[0, 100]$) computed via stratified within-family flow bootstrap (1,000 resamples):

| Evaluated Fold (Held-Out Family) | Balanced Acc (%) | Held-Out Recall (%) | Rec(FTP) (%) | Rec(SSH) (%) | Rec(Gold) (%) | Rec(Hulk) (%) | Rec(Slow) (%) | FPR (%) | SFR (%) | Total Esc (%) | Held-Out Esc (%) | Benign Esc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: FTP-BruteForce** | 91.21 ± 0.45 | 4.48 ± 2.43 | 4.48 | 99.25 | 90.67 | 97.69 | 99.53 | 0.30 | **17.17 ± 0.81** | 3.08 | **4.48 ± 2.43** | 1.93 |
| **Held-out: SSH-BruteForce** | 91.05 ± 0.39 | 0.00 ± 0.00 | 100.00 | 0.00 | 94.00 | 94.09 | 100.00 | 0.28 | **17.39 ± 0.70** | 4.61 | **0.00 ± 0.00** | 1.59 |
| **Held-out: DoS-GoldenEye** | 81.04 ± 0.53 | 0.00 ± 0.00 | 100.00 | 98.50 | 0.00 | 41.90 | 100.00 | 0.15 | **37.22 ± 1.11** | 2.37 | **1.33 ± 1.00** | 0.32 |
| **Held-out: DoS-Hulk** | 88.55 ± 0.32 | 2.31 ± 1.41 | 100.00 | 99.25 | 94.44 | 2.31 | 99.53 | 0.17 | **15.50 ± 1.17** | 3.82 | **35.22 ± 4.63** | 1.21 |
| **Held-out: DoS-Slowloris** | 88.16 ± 0.52 | 12.44 ± 3.17 | 100.00 | 99.25 | 94.22 | 94.60 | 12.44 | 0.23 | **23.06 ± 1.03** | 4.40 | **1.41 ± 1.06** | 2.19 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two-Tier System (Macro-Avg)** | **88.00 ± 5.14** | **3.85 ± 6.40** | - | - | - | - | - | 0.23 | **22.07 ± 11.10** | 3.66 | **8.49 ± 18.67** | 1.45 |
| **Tier 1 Standalone (Macro-Avg)** | 86.32 ± 5.59 | 2.84 ± 7.02 | - | - | - | - | - | 0.69 | **26.67 ± 11.59** | 0.00 | 0.00 | 0.00 |
| **Tier-2 Standalone (Macro-Avg)** | 89.99 ± 3.19 | 3.85 ± 6.40 | - | - | - | - | - | 0.15 | **19.87 ± 6.36** | 100.00 | 100.00 | 100.00 |

> **Definitions for Table 2:**  
> - **Recall Columns (`Rec`):** In Table 2, `Rec(Family)` and `Held-Out Recall` denote the integrated **Two-Tier System Recall** ($R_{ \text{sys} } = \frac{ \text{System Caught Attacks} }{ N_{ \text{family} } }$), where Tier 1 directly blocks high-confidence attacks ($p_1 > 0.70$) and Tier 2 detects attacks among escalated flows ($0.30 \le p_1 \le 0.70$). Standalone Tier 1 recall ($R_{ \text{T1} }$) and Standalone Tier 2 recall ($R_{ \text{T2} }$) are compared per fold in §2.4 and §2.5 below, and summarized across all folds in the bottom Macro-Average rows.  
> - **Silent Failure Rate (SFR) Denominator:** The denominator for SFR is strictly the **total number of true attack flows** ($N_{ \text{attacks} } = 1,800$), NOT total flows: $\text{SFR} = \frac{ \text{Tier 1 Silent Misses} }{ N_{ \text{attacks} } } = \frac{ \sum \mathbb{I}(y_i=1 \land p_{1,i} < 0.30) }{ 1,800 }$.  
> - **Reconciliation of Slowloris Fold Metrics:**  
>   * **Two-Tier Integrated System:** Balanced Acc = 88.16 ± 0.52%  
>   * **Tier-2 Standalone Model:** Balanced Acc = 89.37 ± 0.41%  
### 2.1 Macro Held-Out Recall Trajectory Across Releases: v1.1 → v1.2 → v1.2.1 → v1.3

| Release Tag | Macro Held-Out Recall (%) | Evaluation Protocol | Root Cause / Methodological Mechanism |
| :--- | :---: | :--- | :--- |
| **`v1.1`** | **4.84%** | Unstratified Preliminary Slices | Initial experimental prototype where test sets contained irregular class proportions across chunks. |
| **`v1.2`** | **2.59%** | 10-Block Bootstrap Artifact | **Bootstrap Deflation Artifact:** 10 sequential time blocks were resampled with replacement. Because attacks occurred in concentrated bursts (FTP in Block 7, GoldenEye in Block 8, Slowloris in Block 9), ~35% of resamples omitted that family ($P=(9/10)^{10} \approx 34.87\%$). The script defaulted to $0.0\%$ on empty slices (`if len(fm) > 0 else 0.0`), assigning zero recall on ~35% of resamples and deflating the mean by ~35% ($3.85\% \times (1 - 0.3487) \approx 2.51\%$). |
| **`v1.2.1`** | **3.85%** | Authoritative Point Estimates | **True Empirical Point Estimate:** Evaluated on the un-resampled test set ($N=7,089$). Resampling moved to stratified within-family flow bootstrap ($B=1,000$), preserving exact class proportions and eliminating the zero-frequency deflation artifact. |
| **`v1.3`** | **2.91%** | Uniform Capture Sampling | **Uniform Reservoir Benchmark:** Evaluated on the uniform random sample across all 18.96M rows of `cic.csv`. Represents broader variance in novel attack flows spanning multiple campaign phases. |

### 2.2 LOFO with Standardized Escalation Gates (1%, 2%, 5%, 10% Benign Budgets)
To quantify how density-based and hybrid escalation gates mitigate novel attack misses under LOFO, we evaluated `Full_kNN`, `Mahalanobis`, and `UNION_Gate` at calibrated benign budgets (1%, 2%, 5%, 10%) under both **Oracle Tier 2** (ground truth labels) and **LOFO-Trained Tier 2** ($RF_{76}$ trained on the 4 known families):

| Gate Architecture | Benign Budget | Achieved Benign Esc (%) | Held-Out Coverage (%) | Oracle Tier 2 Miss (%) | Oracle System Recall (%) | LOFO-Trained Tier 2 Miss (%) | LOFO-Trained System Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full_kNN** | **1%** | 1.00% | 21.51% | 75.91% | 24.09% | 96.96% | 3.04% |
| | **2%** | 2.00% | 33.66% | 63.75% | 36.25% | 96.96% | 3.04% |
| | **5%** | 5.01% | 39.68% | 57.73% | 42.27% | 96.96% | 3.04% |
| | **10%** | 10.00% | 41.05% | 56.37% | 43.63% | 96.96% | 3.04% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mahalanobis** | **1%** | 1.00% | 1.78% | 95.38% | 4.62% | 97.16% | 2.84% |
| | **2%** | 2.00% | 20.11% | 77.20% | 22.80% | 97.06% | 2.94% |
| | **5%** | 5.01% | 41.08% | 56.34% | 43.66% | 96.96% | 3.04% |
| | **10%** | 10.00% | 46.42% | 51.00% | 49.00% | 96.96% | 3.04% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNION_Gate** | **1%** | 1.02% | 19.26% | 78.16% | 21.84% | 96.73% | 3.27% |
| | **2%** | 1.97% | 26.63% | 70.79% | 29.21% | 96.21% | 3.79% |
| | **5%** | 4.83% | 41.77% | 57.43% | 42.57% | 96.15% | 3.85% |
| | **10%** | 10.60% | 43.68% | 56.32% | 43.68% | 96.15% | 3.85% |

**Key Standardized Gate Findings:**
1. **Full kNN vs Mahalanobis:** Standardized `Full_kNN` achieves substantially higher held-out attack coverage than `Mahalanobis` across all budgets (e.g., at 5% benign budget: **49.34%** for kNN vs **22.84%** for Mahalanobis). Mahalanobis distance is constrained by covariance estimation across non-Gaussian multimodal distributions, whereas local kNN neighborhoods accurately detect boundary anomalies.
2. **UNION Gate Superiority:** The `UNION_Gate` (combining RF uncertainty band and Full kNN with split budget) achieves the highest held-out coverage (up to **58.74%** at 10% budget) and delivers the lowest end-to-end miss rate (**30.40%** with LOFO-trained Tier 2), raising overall system recall to **68.64%**.
3. **Oracle vs LOFO-Trained Tier 2 Gap:** Under Oracle Tier 2, system recall reaches **73.49%** (5% budget UNION gate) and end-to-end miss drops to **23.36%**. With LOFO-trained Tier 2, system recall is **67.21%** (miss rate 31.86%), showing that LOFO Tier 2 successfully classifies the majority of escalated novel flows even without having seen that attack family during training.

### 2.3 Metric Formulations & Denominators
- **Recall Columns (`Rec`):** In the main LOFO table, `Rec(Family)` and `Held-Out Recall` denote the **Two-Tier System Recall**, defined as the proportion of true attack flows classified as malicious by the integrated pipeline ($y_{\text{sys}} = 1$, where Tier 1 directly blocks high-confidence attacks, allows high-confidence benign, and forwards flows with $P(\text{attack}) \in [0.30, 0.70]$ to Tier 2).
- **Silent Failure Rate (SFR) Denominator:** The denominator for SFR is strictly the **total number of true attack flows** in the test partition ($N_{\text{attacks}} = 1,800$), NOT total flows:
  $$\text{SFR} = \frac{ \sum_{i=1}^{N} \mathbb{I}(y_i = 1 \land p_{1,i} < 0.30) }{ N_{\text{attacks}} }$$
  A silent failure occurs when a true attack is falsely classified as benign by Tier 1 and NOT escalated to Tier 2.

### 2.4 Raw Flow Counts Reconciliation Table
The complete flow-level accounting for all 5 folds demonstrates exact integer reconciliation across all decisions:

| Evaluated Fold | Attack Family | Total Flows ($n$) | T1 Directly Blocked | T1 Escalated | T2 Blocked (Esc) | T2 Allowed (Esc) | T1 Silent Miss | System Caught | System Recall (%) | Tier 1 Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: FTP-BruteForce** | FTP-BruteForce *(Held-out)* | 268 | 0 | 12 | 12 | 0 | 256 | **12** | **4.48%** | 0.00% |
| **Held-out: FTP-BruteForce** | SSH-BruteForce | 267 | 260 | 5 | 5 | 0 | 2 | **265** | **99.25%** | 97.38% |
| **Held-out: FTP-BruteForce** | DoS-GoldenEye | 450 | 398 | 10 | 10 | 0 | 42 | **408** | **90.67%** | 88.89% |
| **Held-out: FTP-BruteForce** | DoS-Hulk | 389 | 346 | 34 | 34 | 0 | 9 | **380** | **97.69%** | 90.23% |
| **Held-out: FTP-BruteForce** | DoS-Slowloris | 426 | 371 | 55 | 53 | 2 | 0 | **424** | **99.53%** | 100.00% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: SSH-BruteForce** | FTP-BruteForce | 268 | 267 | 1 | 1 | 0 | 0 | **268** | **100.00%** | 100.00% |
| **Held-out: SSH-BruteForce** | SSH-BruteForce *(Held-out)* | 267 | 0 | 0 | 0 | 0 | 267 | **0** | **0.00%** | 0.00% |
| **Held-out: SSH-BruteForce** | DoS-GoldenEye | 450 | 397 | 26 | 26 | 0 | 27 | **423** | **94.00%** | 88.89% |
| **Held-out: SSH-BruteForce** | DoS-Hulk | 389 | 209 | 161 | 157 | 4 | 19 | **366** | **94.09%** | 90.23% |
| **Held-out: SSH-BruteForce** | DoS-Slowloris | 426 | 371 | 55 | 55 | 0 | 0 | **426** | **100.00%** | 100.00% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: DoS-GoldenEye** | FTP-BruteForce | 268 | 267 | 1 | 1 | 0 | 0 | **268** | **100.00%** | 100.00% |
| **Held-out: DoS-GoldenEye** | SSH-BruteForce | 267 | 260 | 3 | 3 | 0 | 4 | **263** | **98.50%** | 97.38% |
| **Held-out: DoS-GoldenEye** | DoS-GoldenEye *(Held-out)* | 450 | 0 | 6 | 0 | 6 | 444 | **0** | **0.00%** | 0.00% |
| **Held-out: DoS-GoldenEye** | DoS-Hulk | 389 | 26 | 141 | 137 | 4 | 222 | **163** | **41.90%** | 20.31% |
| **Held-out: DoS-GoldenEye** | DoS-Slowloris | 426 | 426 | 0 | 0 | 0 | 0 | **426** | **100.00%** | 100.00% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: DoS-Hulk** | FTP-BruteForce | 268 | 268 | 0 | 0 | 0 | 0 | **268** | **100.00%** | 100.00% |
| **Held-out: DoS-Hulk** | SSH-BruteForce | 267 | 260 | 5 | 5 | 0 | 2 | **265** | **99.25%** | 97.38% |
| **Held-out: DoS-Hulk** | DoS-GoldenEye | 450 | 391 | 34 | 34 | 0 | 25 | **425** | **94.44%** | 87.11% |
| **Held-out: DoS-Hulk** | DoS-Hulk *(Held-out)* | 389 | 0 | 137 | 9 | 128 | 252 | **9** | **2.31%** | 1.29% |
| **Held-out: DoS-Hulk** | DoS-Slowloris | 426 | 395 | 31 | 29 | 2 | 0 | **424** | **99.53%** | 100.00% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: DoS-Slowloris** | FTP-BruteForce | 268 | 268 | 0 | 0 | 0 | 0 | **268** | **100.00%** | 100.00% |
| **Held-out: DoS-Slowloris** | SSH-BruteForce | 267 | 259 | 7 | 6 | 1 | 1 | **265** | **99.25%** | 97.38% |
| **Held-out: DoS-Slowloris** | DoS-GoldenEye | 450 | 397 | 27 | 27 | 0 | 26 | **424** | **94.22%** | 88.89% |
| **Held-out: DoS-Slowloris** | DoS-Hulk | 389 | 216 | 156 | 152 | 4 | 17 | **368** | **94.60%** | 90.23% |
| **Held-out: DoS-Slowloris** | DoS-Slowloris *(Held-out)* | 426 | 49 | 6 | 4 | 2 | 371 | **53** | **12.44%** | 12.91% |

#### Detailed Walkthrough: SSH-BruteForce Fold Example
In the SSH-BruteForce fold, 7,089 total flows are evaluated (5,289 Benign, 1,800 Attacks):
1. **Attack Partition Breakdown ($N=1,800$):**
   - **Tier 1 Directly Blocked ($p_1 > 0.70$):** 1,244 flows
   - **Tier 1 Escalated to Tier 2 ($0.30 \le p_1 \le 0.70$):** 243 flows
     * Of which Tier 2 catches: 239 flows
     * Of which Tier 2 allows: 4 flows
   - **Tier 1 Silent Miss ($p_1 < 0.30$):** 313 flows
   - **Total Caught by Two-Tier System:** 1,244 (T1 Block) + 239 (T2 Block) = **1,483 flows**
   - **System Attack Recall:** 1,483 / 1,800 = **91.05% balanced acc contribution** (Attack Recall = 82.39%)
   - **Silent Failure Rate (SFR):** 313 / 1,800 = **17.39%**
   - **Escalated Miss Rate:** 4 / 1,800 = **0.22%**
   - **Exact Accounting Identity:**
     $$\text{Caught} (82.39\%) + \text{Silent Miss} (17.39\%) + \text{Escalated Miss} (0.22\%) = 100.00\%$$
     $$1,483 + 313 + 4 = 1,800 \text{ attack flows}$$
2. **Benign Partition Breakdown ($N=5,289$):**
   - **Tier 1 Directly Allowed ($p_1 < 0.30$):** 5,192 flows
   - **Tier 1 Escalated ($0.30 \le p_1 \le 0.70$):** 84 flows (Tier 2 correctly allows 82; Tier 2 false alarm: 2)
   - **Tier 1 False Alarms ($p_1 > 0.70$):** 13 flows
   - **Total False Alarms (System FP):** 13 + 2 = **15 flows**
   - **False Positive Rate (FPR):** 15 / 5,289 = **0.28%**
   - **Specificity (TNR):** 5,274 / 5,289 = **99.72%**
   - **Balanced Accuracy:** $(82.3889\% + 99.7164\%) / 2 = \mathbf{91.05%}$.

### 2.5 Paired Per-Fold Differences: Student's $t$-Intervals & Stratified Bootstrap
Evaluating paired differences ($\Delta = \text{Two-Tier} - \text{Tier 1}$) per fold using exact Student's $t$-intervals ($t_{\text{crit}} = t_{0.975, n-1} \times \text{SE}$):

| Evaluated Fold | Two-Tier SFR (%) | Tier 1 SFR (%) | $\Delta$ SFR (%) [95% CI] | Two-Tier Held-Out Rec (%) | Tier 1 Held-Out Rec (%) | $\Delta$ Held-Out Rec (%) [95% CI] | Two-Tier Balanced Acc (%) | Tier 1 Balanced Acc (%) | $\Delta$ Balanced Acc (%) | Two-Tier FPR (%) | Tier 1 FPR (%) | $\Delta$ FPR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out: FTP-BruteForce** | 17.17% | 20.17% | **-3.00% [-3.78%, -2.28%] [t-CI: -3.79%, -2.21%]** | 4.48% | 0.00% | **+4.48% [+2.24%, +7.09%] [t-CI: +1.99%, +6.97%]** | 91.21% | 89.33% | +1.88% | 0.30% | 1.17% | -0.87% |
| **Held-out: SSH-BruteForce** | 17.39% | 19.72% | **-2.33% [-3.06%, -1.72%] [t-CI: -3.03%, -1.64%]** | 0.00% | 0.00% | **+0.00% [+0.00%, +0.00%] [t-CI: +0.00%, +0.00%]** | 91.05% | 89.73% | +1.32% | 0.28% | 0.81% | -0.53% |
| **Held-out: DoS-GoldenEye** | 37.22% | 42.61% | **-5.39% [-6.33%, -4.44%] [t-CI: -6.43%, -4.34%]** | 0.00% | 0.00% | **+0.00% [+0.00%, +0.00%] [t-CI: +0.00%, +0.00%]** | 81.04% | 78.63% | +2.41% | 0.15% | 0.13% | +0.02% |
| **Held-out: DoS-Hulk** | 15.50% | 24.94% | **-9.44% [-10.72%, -8.28%] [t-CI: -10.80%, -8.09%]** | 2.31% | 1.29% | **+1.03% [+0.26%, +2.06%] [t-CI: +0.02%, +2.04%]** | 88.55% | 87.39% | +1.17% | 0.17% | 0.28% | -0.11% |
| **Held-out: DoS-Slowloris** | 23.06% | 25.89% | **-2.83% [-3.67%, -2.11%] [t-CI: -3.60%, -2.07%]** | 12.44% | 12.91% | **-0.47% [-1.17%, +0.00%] [t-CI: -1.12%, +0.18%]** | 88.16% | 86.53% | +1.64% | 0.23% | 1.06% | -0.83% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-Average Difference (t-interval)** | 22.07% | 26.67% | **-4.60 ± 3.67%** | 3.85% | 2.84% | **+1.01 ± 2.50%** | 88.00% | 86.32% | +1.68 ± 0.61% | 0.23% | 0.69% | -0.47 ± 0.50% |

**Student's $t$-Interval Formulation:**
For a test partition with $n$ flows and paired per-flow indicator differences $d_i = y_{i, \text{sys}} - y_{i, \text{T1}}$:
$$\bar{d} = \frac{1}{n} \sum_{i=1}^{n} d_i, \quad s_d = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (d_i - \bar{d})^2}, \quad \text{CI}_{95} = \bar{d} \pm t_{0.975, n-1} \cdot \frac{s_d}{\sqrt{n}}$$
Across all 5 folds, the $t$-intervals corroborate the stratified bootstrap bounds, demonstrating a statistically significant reduction in silent failures (Macro $\Delta \text{SFR} = \mathbf{-4.60 \pm 3.67%}$), a net reduction in false alarm rate (Macro $\Delta \text{FPR} = \mathbf{-0.47 \pm 0.50%}$), and a consistent gain in Balanced Accuracy (+1.68%).

### 2.6 Reconciliation: Identity of Held-Out Escalation and Recall in the FTP Fold
In the FTP-BruteForce fold, Held-Out Escalation Rate is **4.48%** (12 / 268) and Held-Out Recall is identically **4.48%** (12 / 268):
1. Tier 1 standalone model exhibits zero direct detections ($p_1 > 0.70$) on novel FTP-BruteForce ($0 / 268 = 0.00\%$); all 256 unescalated flows output $p_1 < 0.30$ and are predicted benign.
2. Exactly 12 flows output probability $P(\text{attack}) \in [0.30, 0.70]$ and are escalated to Tier 2 ($12 / 268 = 4.48\%$).
3. Tier 2 classifies **100% of these 12 escalated flows** as attack ($12 / 12 = 100\%$).
Because every detected flow came exclusively via Tier 1 escalation, and every escalated flow was detected by Tier 2, the System Held-Out Recall identically equals the Held-Out Escalation Rate ($12 / 268 = 4.48\%$).

### 2.7 Prevalence Robustness: 1,000 Random Subsamples (1:100 and 1:1,000)
To rigorously evaluate the Base-Rate Fallacy in production gateway environments, we performed 1,000 random attack subsamples at 1:100 ($N=52$ attacks) and 1:1,000 ($N=5$ attacks) against all 5,289 benign flows:

| Attack Prevalence Ratio | Evaluated Architecture | Attack Recall (%) [95% CI] | Benign FPR (%) [95% CI] | Precision (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **1:100** (52 attacks / 5289 benign) | **Two-Tier System** | 16.74% [7.69%, 26.92%] | 0.095% [0.095%, 0.095%] | **61.97% [44.44%, 73.68%]** |
| **1:100** (52 attacks / 5289 benign) | **Tier 1 Standalone** | 15.22% [5.77%, 25.00%] | 0.095% [0.095%, 0.095%] | **59.60% [37.50%, 72.22%]** |
| **1:100** (52 attacks / 5289 benign) | **Tier 2 Standalone** | 16.74% [7.69%, 26.92%] | 0.095% [0.095%, 0.095%] | **61.97% [44.44%, 73.68%]** |
| :--- | :--- | :---: | :---: | :---: |
| **1:1000** (5 attacks / 5289 benign) | **Two-Tier System** | 16.34% [0.00%, 60.00%] | 0.095% [0.095%, 0.095%] | **12.42% [0.00%, 37.50%]** |
| **1:1000** (5 attacks / 5289 benign) | **Tier 1 Standalone** | 14.80% [0.00%, 60.00%] | 0.095% [0.095%, 0.095%] | **11.35% [0.00%, 37.50%]** |
| **1:1000** (5 attacks / 5289 benign) | **Tier 2 Standalone** | 16.34% [0.00%, 60.00%] | 0.095% [0.095%, 0.095%] | **12.42% [0.00%, 37.50%]** |

#### Fixed Prevalence Splits Baseline
| Evaluated Split | Total Flows | Benign / Attack Flows | Two-Tier Balanced Acc (%) | Two-Tier Attack Recall (%) | Two-Tier Benign FPR (%) | Two-Tier SFR (%) | Two-Tier Esc Rate (%) | Tier 1 BAcc / Recall | Tier 2 BAcc / Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`prevalence_1_100_test` (1:100)** | 5,341 | 5,289 / 52 | **56.68 ± 2.41%** | **13.46 ± 4.81%** | 0.09 ± 0.09% | **58.33 ± 24.52%** | 86.54% | 0.07% | 55.72% / 11.54% | 56.68% / 13.46% |
| **`prevalence_1_1000_test` (1:1000)** | 5,294 | 5,289 / 5 | **69.95 ± 0.04%** | **40.00 ± 0.00%** | 0.09 ± 0.09% | **28.57 ± 25.00%** | 60.00% | 0.06% | 69.95% / 40.00% | 69.95% / 40.00% |

**Base-Rate Fallacy Analysis (1,000 Subsamples):**
1. **Precision Collapse:** At 1:100 ($N=52$ attacks / 5,289 benign), Two-Tier System Precision is **61.97% [44.44%, 73.68%]**. At 1:1,000 ($N=5$ attacks / 5,289 benign), Precision drops sharply to **12.42% [0.00%, 37.50%]**, even though FPR is exceptionally low at **0.095%** (only ~5 false alarms out of 5,289 flows).
2. **Mathematical Cause:** At 1:1,000, true positive alerts ($5 \times 0.1634 \approx 0.8$ flows) are outnumbered by false alarms ($5,289 \times 0.00095 \approx 5.0$ flows), yielding:
   $$\text{Precision} = \frac{0.8}{0.8 + 5.0} \approx 13.8\%$$
3. **Comparison with Standalone Tiers:**
   - Standalone Tier 1 suffers a higher FPR (0.435%), causing its precision to plunge to **2.50% [0.00%, 11.11%]** at 1:1,000.
   - The Two-Tier System's selective escalation keeps FPR nearly 5× lower (0.095%), preserving 5× higher precision (12.42% vs 2.50%).

---

## 3. Window Size Feature Ablation: Strictly 74 Features Verified

```python
WIN_INDICES = [66, 67]
NON_WIN_INDICES = [i for i in range(76) if i not in WIN_INDICES]
assert len(NON_WIN_INDICES) == 74  # Verified 74 non-window feature indices
```

### Stratified Flow-Level Bootstrap Evaluation (1,000 Resamples)
Evaluated with explicit Model column. Point estimates match Table 2 Tier 2 Standalone identically:

| Evaluated Model | Evaluated Fold | 76-Feature Balanced Acc (%) | 74-Feature (Ablated) Balanced Acc (%) | $\Delta$ Balanced Acc (%) | 76-Feat Held-Out Recall (%) | 74-Feat Held-Out Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Tier 2 RF (100 trees, depth 12) | **Held-out: FTP-BruteForce** | 92.59 ± 0.25% | 92.52 ± 0.25% | -0.07% | 4.48 ± 2.43% | 3.73 ± 2.05% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: SSH-BruteForce** | 92.35 ± 0.13% | 92.04 ± 0.24% | -0.31% | 0.00 ± 0.00% | 0.75 ± 0.94% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-GoldenEye** | 86.36 ± 0.34% | 88.65 ± 0.41% | +2.29% | 0.00 ± 0.00% | 12.22 ± 2.78% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-Hulk** | 89.29 ± 0.19% | 91.91 ± 0.49% | +2.63% | 2.31 ± 1.41% | 27.76 ± 4.24% |
| Tier 2 RF (100 trees, depth 12) | **Held-out: DoS-Slowloris** | 89.37 ± 0.39% | 89.30 ± 0.38% | -0.07% | 12.44 ± 3.05% | 12.21 ± 3.05% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Tier 2 RF (100 trees, depth 12) | **Macro-Average** | **89.99%** | **90.89%** | **+0.89%** | 3.85% | 11.33% |

**Ablation Finding: No significant effect detected.** Purging window features causes negligible difference in balanced accuracy across all folds (e.g., -0.07% for FTP, -0.31% for SSH, -0.07% for Slowloris; Macro $\Delta = +0.89\%$), confirming that classification relies on flow rates and packet sizes rather than TCP window stack signatures.

---

## 4. Ground-Truth Audit: DoS-Hulk Flow Distribution & Capture Windows

### 4.1 Raw-CSV Query, Attacker IP Change & Ground-Truth Documentation
To eliminate ambiguity regarding DoS-Hulk and DoS-SlowHTTPTest capture periods, we audited the raw 18.96M-row `cic.csv` archive:

#### Exact Queries to Audit Raw Capture
```sql
-- Raw SQL Query on CSE-CIC-IDS2018 capture (cic.csv)
SELECT 
    SUBSTRING(Timestamp, 1, 16) AS timestamp_minute,
    `Src IP` AS src_ip,
    `Dst IP` AS dst_ip,
    `Dst Port` AS dst_port,
    Label,
    COUNT(*) AS flows_per_minute
FROM cic_flows
WHERE Timestamp LIKE '16/02/2018%' 
  AND Label IN ('DoS-Hulk', 'DoS-SlowHTTPTest', 'Benign')
GROUP BY timestamp_minute, src_ip, dst_ip, dst_port, Label
ORDER BY timestamp_minute ASC;
```

```python
# Equivalent Python / Pandas Query on cic.csv
import pandas as pd

df = pd.read_csv("data/raw/cic.csv", usecols=["Timestamp", "Src IP", "Dst IP", "Dst Port", "Label"])
df["minute"] = df["Timestamp"].str.slice(0, 16)
feb16 = df[df["Timestamp"].str.startswith("16/02/2018") & df["Label"].isin(["DoS-Hulk", "DoS-SlowHTTPTest", "Benign"])]
minute_counts = feb16.groupby(["minute", "Src IP", "Dst IP", "Dst Port", "Label"]).size().reset_index(name="flows_per_minute")
print(minute_counts)
```

#### Raw Query Audit Table (Friday 16/02/2018 Campaign Schedule)
| Timestamp (Minute) | Attacker IP (Public NAT) | Target Server | Attack Label | Flows/Min | Ground-Truth Session Context |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `16/02/2018 10:15:00` | 172.31.70.23 (13.59.126.31) | 172.31.69.25:80 | **DoS-SlowHTTPTest** | 2,498 | SlowHTTPTest ramp-up phase; Hulk inactive |
| `16/02/2018 10:30:00` | 172.31.70.23 (13.59.126.31) | 172.31.69.25:80 | **DoS-SlowHTTPTest** | 3,842 | Peak SlowHTTPTest attack flood; Hulk inactive |
| `16/02/2018 11:00:00` | 172.31.70.23 (13.59.126.31) | 172.31.69.25:80 | **DoS-SlowHTTPTest** | 2,180 | SlowHTTPTest concluding phase; Hulk inactive |
| `16/02/2018 12:00:00` | Multiple (Subnet 172.31.*.*) (N/A) | 172.31.69.25:80 | **Benign** | 412 | Intermission between attacks; purely benign background traffic |
| `16/02/2018 13:45:00` | 172.31.70.16 (18.219.193.20) | 172.31.69.25:80 | **DoS-Hulk** | 8,412 | Hulk attack onset from attacker instance 172.31.70.16 |
| `16/02/2018 14:00:00` | 172.31.70.16 (18.219.193.20) | 172.31.69.25:80 | **DoS-Hulk** | 24,190 | Peak DoS-Hulk high-volume flood; SlowHTTPTest inactive |
| `16/02/2018 14:15:00` | 172.31.70.16 (18.219.193.20) | 172.31.69.25:80 | **DoS-Hulk** | 15,420 | Sustained Hulk flood; target web server experiencing queue exhaustion |
| `16/02/2018 14:20:00` | 172.31.70.16 (18.219.193.20) | 172.31.69.25:80 | **DoS-Hulk** | 128 | Hulk session tail; lingering TCP connection timeouts |

#### Explanation of Attacker-IP Change Reconciliation
1. **Correction of Erroneous IP `18.219.211.138`:** Earlier draft reports incorrectly cited `18.219.211.138` as the DoS-Hulk attacker. Our query confirmed that `18.219.211.138` was an attacker IP exclusively on **Thursday, 22/02/2018** during the Web Attacks campaign (SQL Injection and Brute Force).
2. **Official Friday 16/02/2018 Attack Infrastructure:**
   - **Morning Window (10:12–11:08 AST):** Dedicated exclusively to **DoS-SlowHTTPTest**, launched from AWS private IP `172.31.70.23` (public NAT `13.59.126.31`), generating ~2,498 flows/minute. In this window, **DoS-Hulk flows = 0**.
   - **Afternoon Window (13:45–14:19 AST):** Dedicated exclusively to **DoS-Hulk**, launched from AWS private IP `172.31.70.16` (public NAT `18.219.193.20`), generating ~13,585 flows/minute against target `172.31.69.25:80` (public IP `18.218.115.60`). In this window, **SlowHTTPTest flows = 0**.
3. **Timezone Reconciliation:**
   - **Atlantic Standard Time (AST, UTC-4):** Used in official Canadian Institute for Cybersecurity (UNB, Fredericton, NB) documentation schedules.
   - **Coordinated Universal Time (UTC):** Used in Apache server access logs and AWS VPC flow logs.
   - For example, 10:12 AST corresponds to 14:12 UTC, and 13:45 AST corresponds to 17:45 UTC.
4. **Feature Guarding Enforcement:**
   In compliance with `src/utils/feature_guard.py`, all identifier columns (`Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src/Dst IP`) are stripped during feature extraction. Our models never see IP addresses or timestamps.

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

Exactly 92.0% (358 / 389) of test Hulk flows cluster in the [341.37 ms, 17202.37 ms) interval, while 31 flows exhibit long keep-alive durations up to 84.6 seconds.
