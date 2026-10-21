# Escalation Triggers Comparative Evaluation & Two-Tier Security Audit

**Execution Timestamp:** 2026-10-21 21:40:56 UTC  
**Dataset:** CSE-CIC-IDS2018 (Benchmark Splits)  
**Evaluation Seeds:** All metrics report mean ± 95% block-bootstrap confidence intervals over 10 chronological time blocks (1,000 resamples).  

---

## 1. Standardized Escalation Gates Comparison

Features are standardized with `StandardScaler` fitted exclusively on clean benign training flows.
Evaluated across:
1. **Full kNN Gate:** $k=5$, Euclidean distance on standardized 9 features ($N=4,000$ reference set).
2. **Prototype kNN Gate:** $M=32$ cluster centroids from `MiniBatchKMeans` on standardized 9 features.
3. **Mahalanobis Distance Gate:** Distance computed with regularized covariance on standardized 9 features.
4. **RF Probability Band:** Uncertainty ranking based on $|P(\text{attack}) - 0.5|$.
5. **UNION Gate (RF Band + kNN):** Logical OR of RF band and Full kNN with split budget.

### 1.1 Performance at Matched Achieved Benign Escalation

| Gate Architecture | Target Budget | Achieved Benign Esc (%) | SSH Esc ($N=267$) | GoldenEye Esc ($N=450$) | Hulk Esc ($N=389$) | Slowloris Esc ($N=426$) | FTP Esc ($N=268$) | Novel Macro Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full kNN (Standardized)** | **1%** | 1.00 ± 0.44% | 0 (0.0%) | 63 (14.0%) | 100 (25.7%) | 277 (65.0%) | 0 (0.0%) | **26.18 ± 26.65%** |
| | **2%** | 2.00 ± 0.56% | 0 (0.0%) | 187 (41.6%) | 269 (69.2%) | 277 (65.0%) | 0 (0.0%) | **43.93 ± 29.93%** |
| | **5%** | 5.01 ± 0.85% | 0 (0.0%) | 255 (56.7%) | 288 (74.0%) | 277 (65.0%) | 0 (0.0%) | **48.93 ± 33.53%** |
| | **10%** | 10.00 ± 2.54% | 0 (0.0%) | 266 (59.1%) | 306 (78.7%) | 277 (65.0%) | 0 (0.0%) | **50.70 ± 34.32%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prototype kNN (M=32)** | **1%** | 1.02 ± 0.41% | 0 (0.0%) | 62 (13.8%) | 78 (20.1%) | 0 (0.0%) | 0 (0.0%) | **8.46 ± 6.20%** |
| | **2%** | 2.00 ± 0.82% | 0 (0.0%) | 125 (27.8%) | 124 (31.9%) | 278 (65.3%) | 0 (0.0%) | **31.23 ± 25.46%** |
| | **5%** | 5.01 ± 0.98% | 0 (0.0%) | 130 (28.9%) | 186 (47.8%) | 279 (65.5%) | 1 (0.4%) | **35.55 ± 26.02%** |
| | **10%** | 10.00 ± 1.52% | 0 (0.0%) | 265 (58.9%) | 322 (82.8%) | 281 (66.0%) | 1 (0.4%) | **51.91 ± 32.09%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mahalanobis Gate** | **1%** | 1.00 ± 0.51% | 0 (0.0%) | 17 (3.8%) | 20 (5.1%) | 0 (0.0%) | 0 (0.0%) | **2.23 ± 1.65%** |
| | **2%** | 2.00 ± 0.88% | 0 (0.0%) | 124 (27.6%) | 116 (29.8%) | 184 (43.2%) | 0 (0.0%) | **25.14 ± 19.37%** |
| | **5%** | 5.01 ± 0.86% | 0 (0.0%) | 250 (55.6%) | 330 (84.8%) | 277 (65.0%) | 0 (0.0%) | **51.35 ± 33.65%** |
| | **10%** | 10.00 ± 1.87% | 0 (0.0%) | 310 (68.9%) | 382 (98.2%) | 277 (65.0%) | 0 (0.0%) | **58.03 ± 37.17%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF Probability Band** | **1%** | 1.19 ± 0.41% | 3 (1.1%) | 0 (0.0%) | 4 (1.0%) | 332 (77.9%) | 23 (8.6%) | **20.02 ± 23.73%** |
| | **2%** | 2.33 ± 0.68% | 7 (2.6%) | 0 (0.0%) | 4 (1.0%) | 332 (77.9%) | 23 (8.6%) | **20.40 ± 23.11%** |
| | **5%** | 8.49 ± 1.57% | 7 (2.6%) | 0 (0.0%) | 4 (1.0%) | 426 (100.0%) | 26 (9.7%) | **25.91 ± 30.19%** |
| | **10%** | 15.94 ± 3.79% | 7 (2.6%) | 21 (4.7%) | 4 (1.0%) | 426 (100.0%) | 29 (10.8%) | **27.08 ± 30.15%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNION Gate (RF + kNN)** | **1%** | 1.04 ± 0.61% | 2 (0.7%) | 51 (11.3%) | 75 (19.3%) | 332 (77.9%) | 4 (1.5%) | **27.32 ± 29.15%** |
| | **2%** | 2.06 ± 0.80% | 3 (1.1%) | 63 (14.0%) | 104 (26.7%) | 332 (77.9%) | 23 (8.6%) | **29.95 ± 31.58%** |
| | **5%** | 5.82 ± 1.20% | 7 (2.6%) | 247 (54.9%) | 277 (71.2%) | 426 (100.0%) | 26 (9.7%) | **57.18 ± 40.96%** |
| | **10%** | 13.31 ± 1.61% | 7 (2.6%) | 255 (56.7%) | 292 (75.1%) | 426 (100.0%) | 26 (9.7%) | **58.59 ± 38.84%** |

> **Purge of Obsolete Numbers:** All older unstandardized numbers (including 40/277/315/692 and 2.22/7.20/65.02) are permanently replaced by the standardized metrics above. Under standardization, kNN captures 187 GoldenEye flows (41.6%) and 269 Hulk flows (69.2%) at 2% benign budget.

---

## 2. Deployable Calibration Thresholds vs Oracle Thresholds & SSH Check

### 2.1 Calibration Split (Deployable) vs Oracle-Matched Thresholds
Thresholds fit on a 50% held-out training calibration split (deployable in production) compared to oracle-matched thresholds on the test partition:

| Gate & Budget | Deployable Threshold | Deployable Achieved Benign | Oracle Threshold | Oracle Achieved Benign |
| :--- | :---: | :---: | :---: | :---: |
| **kNN 1%** | 2.0827 | 1.04% | 1.6768 | 1.00% |
| **Mahalanobis 1%** | 11.8536 | 1.06% | 11.4368 | 1.00% |
| **kNN 2%** | 1.0962 | 2.48% | 1.2934 | 2.00% |
| **Mahalanobis 2%** | 6.5961 | 2.36% | 6.9648 | 2.00% |
| **kNN 5%** | 0.5596 | 6.54% | 0.7551 | 5.01% |
| **Mahalanobis 5%** | 4.0268 | 6.58% | 4.0474 | 5.01% |
| **kNN 10%** | 0.3471 | 13.16% | 0.5347 | 10.00% |
| **Mahalanobis 10%** | 3.7012 | 11.33% | 3.7651 | 10.00% |

### 2.2 Paired Per-Block Differences vs RF Band (10 Chronological Time Blocks)
Paired block differences ($\Delta = \text{Gate} - \text{RF Band}$) across 10 chronological blocks:

| Evaluated Gate vs RF Band | Budget | Mean Paired Difference ($\Delta$) | Blocks With Positive Gain (%) |
| :--- | :---: | :---: | :---: |
| **Full kNN (Standardized) vs RF Band** | **1%** | +1.40 ± 2.06% | 20% |
| **Full kNN (Standardized) vs RF Band** | **2%** | +5.66 ± 7.79% | 20% |
| **Full kNN (Standardized) vs RF Band** | **5%** | +5.59 ± 8.18% | 20% |
| **Full kNN (Standardized) vs RF Band** | **10%** | +6.01 ± 8.75% | 20% |
| **Prototype kNN (M=32) vs RF Band** | **1%** | -2.29 ± 5.27% | 10% |
| **Prototype kNN (M=32) vs RF Band** | **2%** | +2.11 ± 3.47% | 20% |
| **Prototype kNN (M=32) vs RF Band** | **5%** | +2.55 ± 4.56% | 20% |
| **Prototype kNN (M=32) vs RF Band** | **10%** | +6.37 ± 9.10% | 20% |
| **Mahalanobis Gate vs RF Band** | **1%** | -3.70 ± 6.73% | 10% |
| **Mahalanobis Gate vs RF Band** | **2%** | +0.87 ± 2.89% | 10% |
| **Mahalanobis Gate vs RF Band** | **5%** | +6.51 ± 9.24% | 20% |
| **Mahalanobis Gate vs RF Band** | **10%** | +8.01 ± 11.16% | 20% |
| **UNION Gate (RF + kNN) vs RF Band** | **1%** | +1.57 ± 2.28% | 20% |
| **UNION Gate (RF + kNN) vs RF Band** | **2%** | +2.02 ± 3.21% | 20% |
| **UNION Gate (RF + kNN) vs RF Band** | **5%** | +7.40 ± 9.17% | 20% |
| **UNION Gate (RF + kNN) vs RF Band** | **10%** | +7.76 ± 9.62% | 20% |

### 2.3 SSH Check with Standardized 25-Feature Gate
Evaluated on 25 allowlisted features to investigate SSH-BruteForce escalation:

| Target Budget | 25-Feature kNN Escalated | 25-Feature Mahalanobis Escalated |
| :---: | :---: | :---: |
| **1%** | 0 / 267 (0.00%) | 0 / 267 (0.00%) |
| **2%** | 0 / 267 (0.00%) | 0 / 267 (0.00%) |
| **5%** | 0 / 267 (0.00%) | 0 / 267 (0.00%) |
| **10%** | 0 / 267 (0.00%) | 0 / 267 (0.00%) |

- **Distribution Analysis:**
  * Benign 25-Feature kNN Distance: Median = 0.1661, 90th percentile = 1.6772, 99th percentile = 4.0441.
  * SSH-BruteForce 25-Feature kNN Distance: Median = 0.6113, Max = 0.6577.
- **Finding:** SSH-BruteForce flows consist of standard interactive login packets (15–22 packets, length 60–150 B) whose distance from benign centers (max 0.658) falls strictly within the dense benign core ($< 1.677$, 90th percentile). Density gates therefore cannot escalate SSH without exceeding tolerable benign budgets; detection requires stateful credential-rate monitoring.

---

## 3. End-to-End Miss Table & Trained Tier 2 Per-Flow Prediction Audit

We logged per-flow predictions of Trained Tier 2 ($RF_{76}$) on flows escalated by each gate:

| Gate Architecture | Target Budget | Escalated Novel Flows | Caught by Trained T2 | Missed by Trained T2 | Oracle End-to-End Miss | Trained T2 End-to-End Miss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full kNN (Standardized)** | **1%** | 440 | 0 | 440 | 1085 (70.82%) | 1525 (99.54%) |
| **Full kNN (Standardized)** | **2%** | 733 | 0 | 733 | 792 (51.70%) | 1525 (99.54%) |
| **Full kNN (Standardized)** | **5%** | 820 | 0 | 820 | 705 (46.02%) | 1525 (99.54%) |
| **Full kNN (Standardized)** | **10%** | 849 | 0 | 849 | 676 (44.13%) | 1525 (99.54%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prototype kNN (M=32)** | **1%** | 140 | 0 | 140 | 1385 (90.40%) | 1525 (99.54%) |
| **Prototype kNN (M=32)** | **2%** | 527 | 0 | 527 | 998 (65.14%) | 1525 (99.54%) |
| **Prototype kNN (M=32)** | **5%** | 595 | 0 | 595 | 930 (60.70%) | 1525 (99.54%) |
| **Prototype kNN (M=32)** | **10%** | 868 | 0 | 868 | 657 (42.89%) | 1525 (99.54%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mahalanobis Gate** | **1%** | 37 | 0 | 37 | 1488 (97.13%) | 1525 (99.54%) |
| **Mahalanobis Gate** | **2%** | 424 | 0 | 424 | 1101 (71.87%) | 1525 (99.54%) |
| **Mahalanobis Gate** | **5%** | 857 | 0 | 857 | 668 (43.60%) | 1525 (99.54%) |
| **Mahalanobis Gate** | **10%** | 969 | 0 | 969 | 556 (36.29%) | 1525 (99.54%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNION Gate (RF + kNN)** | **1%** | 460 | 39 | 421 | 1072 (69.97%) | 1493 (97.45%) |
| **UNION Gate (RF + kNN)** | **2%** | 502 | 39 | 463 | 1030 (67.23%) | 1493 (97.45%) |
| **UNION Gate (RF + kNN)** | **5%** | 957 | 39 | 918 | 575 (37.53%) | 1493 (97.45%) |
| **UNION Gate (RF + kNN)** | **10%** | 980 | 39 | 941 | 552 (36.03%) | 1493 (97.45%) |

### 3.1 Explanation of Zero Intercept on Pure Density Gates
1. **Orthogonal Trigger Spaces:** Pure kNN and Mahalanobis escalate flows exhibiting anomalous duration and packet spacing (Slowloris, GoldenEye, Hulk).
2. **Discriminative Shortcut:** Trained Tier 2 was trained exclusively on `FTP-BruteForce`, where attacks exhibit high byte volumes and specific packet profiles. Because escalated Slowloris and keep-alive flows have small packet sizes, Tier 2 classifies them as benign.
3. **UNION Gate Intercept:** The **UNION Gate** captures 39 novel flows with Trained Tier 2 because its RF Band component escalates flows with uncertainty in classification space.

### 3.2 Reconciliation of Flow Count Discrepancies
- **0 flows:** Caught by Trained Tier 2 on test Slowloris flows escalated by the 2% kNN gate.
- **1.64%:** Standalone Tier 1 recall on test Slowloris.
- **8.69%:** Standalone Trained Tier 2 recall on test Slowloris.

---

## 4. Algorithm 2 Re-Run with Standardized Gates

Evaluated across 5 random seeds over 5 streaming adaptation cycles.  
**Gate Policy:** Static standardized kNN gate (scaler, reference set, and threshold are NOT refit each cycle to prevent contamination)

| Adaptation Configuration | Final SSH Recall (%) | Final GoldenEye Recall (%) | Final Hulk Recall (%) | Final Slowloris Recall (%) | Final FTP Recall (%) | Final Benign FPR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L=inf, With Rehearsal** | 0.0 ± 0.0% | 54.9 ± 0.0% | 72.6 ± 2.0% | 77.6 ± 0.6% | 100.0 ± 0.0% | 0.01 ± 0.01% |
| **L=inf, No Rehearsal** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 4.0% | 100.0 ± 0.0% | 0.09 ± 0.00% |
| **L=50, With Rehearsal** | 0.0 ± 0.0% | 54.6 ± 0.4% | 43.1 ± 7.9% | 75.1 ± 4.0% | 100.0 ± 0.0% | 0.13 ± 0.02% |
| **L=50, No Rehearsal** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 4.0% | 100.0 ± 0.0% | 0.09 ± 0.00% |

---

## 5. Feedback Poisoning: Unified 25-Seed Evaluation

Conducted across 25 random seeds (`seeds = 100..124`) using CSE-CIC-IDS2018 (`lofo_FTP-BruteForce_train` partition: $N=8,625$ flows; 446 validation attack flows, 446 test attack flows). Injected flows are true attack flows labeled as benign.

- **Target Recall ($R_{\text{target}}$):** Percentage of test attack flows correctly identified as malicious by candidate model.
- **Attack Success ($S_{\text{attack}}$):** 100% - Target Recall (percentage of test attacks evading detection as false negatives).
- **Rejection Policy Note:** In single-cycle adaptation, rejections are 0% below 3.0% rate and peak at ~16% at 5.0%; in 5-cycle adaptation, average cycle rejections reach ~65% at 5%-10% rate. Rejection is NOT unconditionally 100%.

### 5.1 Single-Cycle Poisoning Sweep (25 Seeds, Clean Row Included)

| Injection Rate (%) | Gate Rejection Rate (%) | $\Delta$ Val Acc (%) | $\Delta$ Historical Recall (%) | Candidate Target Recall (%) | Attack Success Without Cap (%) | Attack Success With Cap (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0%** | 0.0% | 0.00 ± 0.00% | 0.00 ± 0.00% | 91.11 ± 0.54% | 8.89 ± 0.54% | **8.89 ± 0.54%** |
| **0.5%** | 0.0% | 0.08 ± 0.05% | 0.16 ± 0.10% | 91.01 ± 0.54% | 8.99 ± 0.54% | **8.88 ± 0.55%** |
| **1.0%** | 0.0% | 0.09 ± 0.05% | 0.25 ± 0.12% | 90.63 ± 0.61% | 9.37 ± 0.61% | **9.21 ± 0.59%** |
| **1.5%** | 0.0% | 0.09 ± 0.06% | 0.20 ± 0.14% | 90.53 ± 0.57% | 9.47 ± 0.57% | **9.14 ± 0.51%** |
| **2.0%** | 0.0% | 0.10 ± 0.05% | 0.39 ± 0.17% | 90.22 ± 0.61% | 9.78 ± 0.61% | **9.07 ± 0.60%** |
| **3.0%** | 4.0% | 0.18 ± 0.08% | 0.59 ± 0.25% | 90.01 ± 0.61% | 9.99 ± 0.61% | **9.18 ± 0.61%** |
| **5.0%** | 16.0% | 0.24 ± 0.10% | 1.00 ± 0.33% | 89.02 ± 0.65% | 10.98 ± 0.65% | **9.66 ± 0.61%** |
| **7.5%** | 12.0% | 0.25 ± 0.09% | 1.01 ± 0.27% | 89.00 ± 0.63% | 11.00 ± 0.63% | **9.50 ± 0.56%** |
| **10.0%** | 12.0% | 0.27 ± 0.10% | 1.01 ± 0.31% | 89.19 ± 0.64% | 10.81 ± 0.64% | **9.60 ± 0.55%** |

### 5.2 Multi-Cycle (5 Consecutive Cycles) Poisoning Sweep (25 Seeds)

| Injection Rate (%) | Rejections Without Cap (%) | Target Recall Without Cap (%) | Attack Success Without Cap (%) | Rejections With Cap (%) | Target Recall With Cap (%) | Attack Success With Cap (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0%** | 0.0 ± 0.0% | 91.11 ± 0.54% | 8.89 ± 0.54% | 0.0 ± 0.0% | 91.11 ± 0.54% | **8.89 ± 0.54%** |
| **0.5%** | 6.4 ± 7.6% | 90.30 ± 0.63% | 9.70 ± 0.63% | 0.0 ± 0.0% | 90.74 ± 0.54% | **9.26 ± 0.54%** |
| **1.0%** | 2.4 ± 3.6% | 89.11 ± 0.63% | 10.89 ± 0.63% | 1.6 ± 2.4% | 90.31 ± 0.56% | **9.69 ± 0.56%** |
| **1.5%** | 4.0 ± 6.0% | 88.30 ± 0.68% | 11.70 ± 0.68% | 0.8 ± 1.2% | 90.28 ± 0.58% | **9.72 ± 0.58%** |
| **2.0%** | 9.6 ± 10.0% | 87.38 ± 0.87% | 12.62 ± 0.87% | 0.0 ± 0.0% | 89.98 ± 0.49% | **10.02 ± 0.49%** |
| **3.0%** | 20.8 ± 8.8% | 85.26 ± 1.33% | 14.74 ± 1.33% | 4.8 ± 6.8% | 89.70 ± 0.59% | **10.30 ± 0.59%** |
| **5.0%** | 64.8 ± 8.0% | 87.30 ± 1.38% | 12.70 ± 1.38% | 4.0 ± 6.0% | 89.01 ± 0.67% | **10.99 ± 0.67%** |
| **7.5%** | 64.8 ± 7.2% | 87.18 ± 1.27% | 12.82 ± 1.27% | 5.6 ± 7.6% | 88.96 ± 0.65% | **11.04 ± 0.65%** |
| **10.0%** | 63.2 ± 7.2% | 86.68 ± 1.54% | 13.32 ± 1.54% | 4.0 ± 6.0% | 89.02 ± 0.64% | **10.98 ± 0.64%** |

### 5.3 Technical Reconciliation
1. **Gate Tolerance:** In `src/feedback/production_loop.py`, `RetrainingGate` enforces `max_acc_drop = 0.02` (2.00%). When regression exceeds 2.00%, the candidate model is rejected and rolled back.
2. **Cap Effectiveness:** Under 5 cycles of poisoning, `apply_poisoning_bound` (20% influence cap) curbs attack success significantly across all injection rates.
