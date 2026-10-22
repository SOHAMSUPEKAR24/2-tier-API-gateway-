# Escalation Triggers Comparative Evaluation & Two-Tier Security Audit

**Execution Timestamp:** 2026-10-22 06:21:07 UTC  
**Dataset:** CSE-CIC-IDS2018 (Benchmark Splits)  
**Release Tag:** `v1.2` (Newest Authoritative Benchmark Freeze)  
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

| Gate Architecture | Target Budget | Achieved Benign Esc (%) | SSH Esc ($N=267$) | GoldenEye Esc ($N=450$) | Hulk Esc ($N=389$) | Slowloris Esc ($N=426$) | FTP Esc ($N=268$) | Novel Macro Coverage (%) | Novel Pooled Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full kNN (Standardized)** | **1%** | 1.00 ± 0.44% | 0 (0.0%) | 63 (14.0%) | 100 (25.7%) | 277 (65.0%) | 0 (0.0%) | **26.18 ± 26.65%** | **28.72 ± 32.26%** |
| | **2%** | 2.00 ± 0.56% | 0 (0.0%) | 187 (41.6%) | 269 (69.2%) | 277 (65.0%) | 0 (0.0%) | **43.93 ± 29.93%** | **47.85 ± 34.97%** |
| | **5%** | 5.01 ± 0.85% | 0 (0.0%) | 255 (56.7%) | 288 (74.0%) | 277 (65.0%) | 0 (0.0%) | **48.93 ± 33.53%** | **53.52 ± 38.24%** |
| | **10%** | 10.00 ± 2.54% | 0 (0.0%) | 266 (59.1%) | 306 (78.7%) | 277 (65.0%) | 0 (0.0%) | **50.70 ± 34.32%** | **55.42 ± 38.36%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prototype kNN (M=32)** | **1%** | 1.00 ± 0.51% | 0 (0.0%) | 125 (27.8%) | 124 (31.9%) | 278 (65.3%) | 0 (0.0%) | **31.23 ± 26.88%** | **34.40 ± 32.02%** |
| | **2%** | 2.00 ± 0.82% | 0 (0.0%) | 125 (27.8%) | 148 (38.0%) | 279 (65.5%) | 1 (0.4%) | **32.83 ± 24.85%** | **36.03 ± 30.28%** |
| | **5%** | 5.01 ± 1.31% | 0 (0.0%) | 255 (56.7%) | 289 (74.3%) | 279 (65.5%) | 1 (0.4%) | **49.11 ± 31.66%** | **53.72 ± 36.31%** |
| | **10%** | 10.00 ± 1.98% | 0 (0.0%) | 270 (60.0%) | 314 (80.7%) | 280 (65.7%) | 1 (0.4%) | **51.61 ± 31.63%** | **56.40 ± 35.85%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mahalanobis Gate** | **1%** | 1.00 ± 0.51% | 0 (0.0%) | 17 (3.8%) | 20 (5.1%) | 0 (0.0%) | 0 (0.0%) | **2.23 ± 1.65%** | **2.42 ± 1.46%** |
| | **2%** | 2.00 ± 0.88% | 0 (0.0%) | 124 (27.6%) | 116 (29.8%) | 184 (43.2%) | 0 (0.0%) | **25.14 ± 19.37%** | **27.68 ± 22.18%** |
| | **5%** | 5.01 ± 0.86% | 0 (0.0%) | 250 (55.6%) | 330 (84.8%) | 277 (65.0%) | 0 (0.0%) | **51.35 ± 33.65%** | **55.94 ± 37.72%** |
| | **10%** | 10.00 ± 1.87% | 0 (0.0%) | 310 (68.9%) | 382 (98.2%) | 277 (65.0%) | 0 (0.0%) | **58.03 ± 37.17%** | **63.25 ± 40.65%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF Probability Band** | **1%** | 1.19 ± 0.41% | 3 (1.1%) | 0 (0.0%) | 4 (1.0%) | 332 (77.9%) | 23 (8.6%) | **20.02 ± 23.73%** | **22.13 ± 31.89%** |
| | **2%** | 2.33 ± 0.68% | 7 (2.6%) | 0 (0.0%) | 4 (1.0%) | 332 (77.9%) | 23 (8.6%) | **20.40 ± 23.11%** | **22.39 ± 31.63%** |
| | **5%** | 8.49 ± 1.57% | 7 (2.6%) | 0 (0.0%) | 4 (1.0%) | 426 (100.0%) | 26 (9.7%) | **25.91 ± 30.19%** | **28.52 ± 40.23%** |
| | **10%** | 15.94 ± 3.79% | 7 (2.6%) | 21 (4.7%) | 4 (1.0%) | 426 (100.0%) | 29 (10.8%) | **27.08 ± 30.15%** | **29.90 ± 39.79%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNION Gate (RF + kNN)** | **1%** | 1.04 ± 0.61% | 2 (0.7%) | 51 (11.3%) | 75 (19.3%) | 332 (77.9%) | 4 (1.5%) | **27.32 ± 29.15%** | **30.03 ± 35.61%** |
| | **2%** | 2.06 ± 0.80% | 3 (1.1%) | 63 (14.0%) | 104 (26.7%) | 332 (77.9%) | 23 (8.6%) | **29.95 ± 31.58%** | **32.77 ± 37.98%** |
| | **5%** | 5.82 ± 1.20% | 7 (2.6%) | 247 (54.9%) | 277 (71.2%) | 426 (100.0%) | 26 (9.7%) | **57.18 ± 40.96%** | **62.47 ± 47.75%** |
| | **10%** | 13.31 ± 1.61% | 7 (2.6%) | 255 (56.7%) | 292 (75.1%) | 426 (100.0%) | 26 (9.7%) | **58.59 ± 38.84%** | **63.97 ± 46.58%** |

> **Purge of Obsolete Numbers:** All older unstandardized preliminary metrics are permanently purged and superseded by the standardized metrics above. Under standardization, kNN captures 187 GoldenEye flows (41.6%) and 269 Hulk flows (69.2%) at 2% benign budget.

---

## 2. Deployable Calibration Thresholds vs Oracle Thresholds & SSH Check

### 2.1 Calibration Split (Deployable) vs Oracle-Matched Thresholds
Thresholds fit on a 50% held-out training calibration split (deployable in production) compared to oracle-matched thresholds on the test partition, reporting achieved benign escalation, novel macro coverage, and novel pooled coverage:

| Gate & Budget | Deployable Threshold | Deployable Achieved Benign | Deployable Macro Coverage (%) | Deployable Pooled Coverage (%) | Oracle Threshold | Oracle Achieved Benign | Oracle Macro Coverage (%) | Oracle Pooled Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **kNN 1%** | 2.0827 | 1.04% | 31.30% | 34.46% | 1.6768 | 1.00% | 26.18% | 28.72% |
| **Mahalanobis 1%** | 11.8536 | 1.06% | 9.62% | 10.64% | 11.4368 | 1.00% | 2.23% | 2.42% |
| **kNN 2%** | 1.0962 | 2.48% | 48.48% | 53.00% | 1.2934 | 2.00% | 43.93% | 47.85% |
| **Mahalanobis 2%** | 6.5961 | 2.36% | 31.75% | 34.92% | 6.9648 | 2.00% | 25.14% | 27.68% |
| **kNN 5%** | 0.5596 | 6.54% | 52.48% | 57.31% | 0.7551 | 5.01% | 48.93% | 53.52% |
| **Mahalanobis 5%** | 4.0268 | 6.58% | 51.58% | 56.20% | 4.0474 | 5.01% | 51.35% | 55.94% |
| **kNN 10%** | 0.3471 | 13.16% | 56.22% | 61.62% | 0.5347 | 10.00% | 50.70% | 55.42% |
| **Mahalanobis 10%** | 3.7012 | 11.33% | 58.19% | 63.45% | 3.7651 | 10.00% | 58.03% | 63.25% |

### 2.2 Paired Per-Block Differences vs RF Band (10 Chronological Time Blocks)
Paired block differences ($\Delta = \text{Evaluated Gate} - \text{Baseline RF Band}$) evaluated across 10 chronological blocks (standardized 9 features, baseline = RF Probability Band uncertainty gate):

| Evaluated Gate Architecture | Target Budget | Mean $\Delta$ Macro Coverage (%) | Blocks Positive (Macro) | Mean $\Delta$ Pooled Coverage (%) | Blocks Positive (Pooled) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full kNN (Standardized)** (Full kNN Gate (Standardized, k=5, 9 features, N=4000 reference)) | **1%** | +1.40 ± 2.06% | 20% | +1.26 ± 2.02% | 20% |
| **Full kNN (Standardized)** (Full kNN Gate (Standardized, k=5, 9 features, N=4000 reference)) | **2%** | +5.66 ± 7.79% | 20% | +5.08 ± 7.25% | 20% |
| **Full kNN (Standardized)** (Full kNN Gate (Standardized, k=5, 9 features, N=4000 reference)) | **5%** | +5.59 ± 8.18% | 20% | +5.00 ± 8.49% | 20% |
| **Full kNN (Standardized)** (Full kNN Gate (Standardized, k=5, 9 features, N=4000 reference)) | **10%** | +6.01 ± 8.75% | 20% | +5.11 ± 8.53% | 20% |
| **Prototype kNN (M=32)** (Prototype kNN Gate (M=32 cluster centroids, 9 features)) | **1%** | +2.42 ± 3.33% | 20% | +2.49 ± 3.83% | 20% |
| **Prototype kNN (M=32)** (Prototype kNN Gate (M=32 cluster centroids, 9 features)) | **2%** | +2.77 ± 4.32% | 20% | +2.54 ± 4.37% | 20% |
| **Prototype kNN (M=32)** (Prototype kNN Gate (M=32 cluster centroids, 9 features)) | **5%** | +5.65 ± 8.25% | 20% | +5.04 ± 8.52% | 20% |
| **Prototype kNN (M=32)** (Prototype kNN Gate (M=32 cluster centroids, 9 features)) | **10%** | +6.24 ± 9.01% | 20% | +5.32 ± 8.72% | 20% |
| **Mahalanobis Gate** (Mahalanobis Distance Gate (Regularized covariance, 9 features)) | **1%** | -3.70 ± 6.73% | 10% | -4.37 ± 8.22% | 10% |
| **Mahalanobis Gate** (Mahalanobis Distance Gate (Regularized covariance, 9 features)) | **2%** | +0.87 ± 2.89% | 10% | +0.76 ± 4.15% | 10% |
| **Mahalanobis Gate** (Mahalanobis Distance Gate (Regularized covariance, 9 features)) | **5%** | +6.51 ± 9.24% | 20% | +5.52 ± 8.83% | 20% |
| **Mahalanobis Gate** (Mahalanobis Distance Gate (Regularized covariance, 9 features)) | **10%** | +8.01 ± 11.16% | 20% | +6.80 ± 10.45% | 20% |
| **UNION Gate (RF + kNN)** (UNION Gate (RF Band + Full kNN, split budget, 9 features)) | **1%** | +1.57 ± 2.28% | 20% | +1.62 ± 2.17% | 20% |
| **UNION Gate (RF + kNN)** (UNION Gate (RF Band + Full kNN, split budget, 9 features)) | **2%** | +2.02 ± 3.21% | 20% | +1.99 ± 2.98% | 20% |
| **UNION Gate (RF + kNN)** (UNION Gate (RF Band + Full kNN, split budget, 9 features)) | **5%** | +7.40 ± 9.17% | 20% | +7.31 ± 9.19% | 20% |
| **UNION Gate (RF + kNN)** (UNION Gate (RF Band + Full kNN, split budget, 9 features)) | **10%** | +7.76 ± 9.62% | 20% | +7.33 ± 9.21% | 20% |

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

We logged per-flow predictions of Trained Tier 2 ($RF_{76}$) on flows escalated by each gate, comparing **FTP-only Trained Tier 2**, **LOFO-Trained Tier 2**, and **Oracle Tier 2**:

| Gate Architecture | Target Budget | Escalated Novel Flows | Caught (FTP-T2) | Miss Rate (FTP-T2) | Caught (LOFO-T2) | Miss Rate (LOFO-T2) | Oracle Miss Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full kNN (Standardized)** | **1%** | 440 | 0 | 1525 (99.54%) | 10 | 1515 (98.89%) | 1085 (70.82%) |
| **Full kNN (Standardized)** | **2%** | 733 | 0 | 1525 (99.54%) | 13 | 1512 (98.69%) | 792 (51.70%) |
| **Full kNN (Standardized)** | **5%** | 820 | 0 | 1525 (99.54%) | 13 | 1512 (98.69%) | 705 (46.02%) |
| **Full kNN (Standardized)** | **10%** | 849 | 0 | 1525 (99.54%) | 13 | 1512 (98.69%) | 676 (44.13%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prototype kNN (M=32)** | **1%** | 527 | 0 | 1525 (99.54%) | 9 | 1516 (98.96%) | 998 (65.14%) |
| **Prototype kNN (M=32)** | **2%** | 552 | 0 | 1525 (99.54%) | 13 | 1512 (98.69%) | 973 (63.51%) |
| **Prototype kNN (M=32)** | **5%** | 823 | 0 | 1525 (99.54%) | 14 | 1511 (98.63%) | 702 (45.82%) |
| **Prototype kNN (M=32)** | **10%** | 864 | 0 | 1525 (99.54%) | 15 | 1510 (98.56%) | 661 (43.15%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mahalanobis Gate** | **1%** | 37 | 0 | 1525 (99.54%) | 3 | 1522 (99.35%) | 1488 (97.13%) |
| **Mahalanobis Gate** | **2%** | 424 | 0 | 1525 (99.54%) | 11 | 1514 (98.83%) | 1101 (71.87%) |
| **Mahalanobis Gate** | **5%** | 857 | 0 | 1525 (99.54%) | 13 | 1512 (98.69%) | 668 (43.60%) |
| **Mahalanobis Gate** | **10%** | 969 | 0 | 1525 (99.54%) | 13 | 1512 (98.69%) | 556 (36.29%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNION Gate (RF + kNN)** | **1%** | 460 | 39 | 1493 (97.45%) | 60 | 1472 (96.08%) | 1072 (69.97%) |
| **UNION Gate (RF + kNN)** | **2%** | 502 | 39 | 1493 (97.45%) | 64 | 1468 (95.82%) | 1030 (67.23%) |
| **UNION Gate (RF + kNN)** | **5%** | 957 | 39 | 1493 (97.45%) | 67 | 1465 (95.63%) | 575 (37.53%) |
| **UNION Gate (RF + kNN)** | **10%** | 980 | 39 | 1493 (97.45%) | 67 | 1465 (95.63%) | 552 (36.03%) |

### 3.1 Explanation of Zero Intercept on Pure Density Gates
1. **Orthogonal Trigger Spaces:** Pure kNN and Mahalanobis escalate flows exhibiting anomalous duration and packet spacing (Slowloris, GoldenEye, Hulk).
2. **Discriminative Shortcut:** Trained Tier 2 was trained exclusively on `FTP-BruteForce`, where attacks exhibit high byte volumes and specific packet profiles. Because escalated Slowloris and keep-alive flows have small packet sizes, Tier 2 classifies them as benign.
3. **LOFO-Trained Tier 2 Generalization:** When Tier 2 is trained under LOFO (learning the other 4 attack families), it catches escalated novel flows (up to 13 under kNN/Mahalanobis, and up to 67 under UNION gate), lowering end-to-end miss rates.
4. **UNION Gate Intercept:** The **UNION Gate** captures novel flows with Trained Tier 2 because its RF Band component escalates flows with uncertainty in classification space.

### 3.2 Reconciliation of Flow Count Discrepancies
- **0 flows:** Caught by Trained Tier 2 on test Slowloris flows escalated by the 2% kNN gate.
- **1.64%:** Standalone Tier 1 recall on test Slowloris.
- **8.69%:** Standalone Trained Tier 2 recall on test Slowloris.

---

## 4. Algorithm 2 Re-Run with Standardized Gates

Evaluated across 5 random seeds over 5 streaming adaptation cycles with 10-block bootstrap 95% confidence intervals.  
**Gate Policy:** Static standardized kNN gate (scaler, reference set, and threshold are NOT refit each cycle to prevent contamination)

### 4.1 Final Adaptation Summary Across Arms (Cycle 5)

| Adaptation Configuration | Label Source | Retraining Gate Policy | Final SSH Recall (%) | Final GoldenEye Recall (%) | Final Hulk Recall (%) | Final Slowloris Recall (%) | Final FTP Recall (%) | Final Benign FPR (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Tier 2, L=inf, With Rehearsal** | Oracle | Gate ON (Rollback) | 0.0 ± 0.0% | 54.9 ± 0.0% | 72.6 ± 6.2% | 77.6 ± 0.0% | 100.0 ± 0.0% | 0.01 ± 0.01% |
| **Oracle Tier 2, L=inf, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=inf, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | 54.2 ± 1.2% | 95.6 ± 0.0% | 98.9 ± 0.7% | 87.1 ± 0.0% | 36.7 ± 0.0% | 34.58 ± 2.62% |
| **Oracle Tier 2, L=50, With Rehearsal** | Oracle | Gate ON (Rollback) | 0.0 ± 0.0% | 54.6 ± 0.0% | 43.1 ± 20.2% | 75.1 ± 0.0% | 100.0 ± 0.0% | 0.13 ± 0.14% |
| **Oracle Tier 2, L=50, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=50, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | 99.0 ± 0.7% | 98.1 ± 0.0% | 99.8 ± 0.1% | 90.4 ± 0.0% | 98.2 ± 0.0% | 83.19 ± 2.27% |
| **Trained Tier 2 (Pseudo-Labels), L=inf, With Rehearsal** | Trained Tier 2 (Pseudo-Labels) | Gate ON (Rollback) | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |

### 4.2 Full Cycle-by-Cycle Adaptation Trajectory (Cycles 0 to 5)

| Adaptation Configuration | Label Source | Retraining Gate Policy | Cycle | SSH Recall (%) | GoldenEye Recall (%) | Hulk Recall (%) | Slowloris Recall (%) | FTP Recall (%) | Benign FPR (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Tier 2, L=inf, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 0** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=inf, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 1** | 0.0 ± 0.0% | 54.9 ± 0.0% | 66.3 ± 9.7% | 73.3 ± 0.0% | 100.0 ± 0.0% | 0.08 ± 0.06% |
| **Oracle Tier 2, L=inf, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 2** | 0.0 ± 0.0% | 54.9 ± 0.0% | 69.0 ± 8.2% | 75.4 ± 0.0% | 100.0 ± 0.0% | 0.06 ± 0.06% |
| **Oracle Tier 2, L=inf, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 3** | 0.0 ± 0.0% | 54.9 ± 0.0% | 70.4 ± 7.2% | 75.4 ± 0.0% | 100.0 ± 0.0% | 0.02 ± 0.01% |
| **Oracle Tier 2, L=inf, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 4** | 0.0 ± 0.0% | 54.9 ± 0.0% | 71.8 ± 8.7% | 77.6 ± 0.0% | 100.0 ± 0.0% | 0.03 ± 0.03% |
| **Oracle Tier 2, L=inf, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 5** | 0.0 ± 0.0% | 54.9 ± 0.0% | 72.6 ± 6.2% | 77.6 ± 0.0% | 100.0 ± 0.0% | 0.01 ± 0.01% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Tier 2, L=inf, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 0** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=inf, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 1** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=inf, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 2** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=inf, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 3** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.12% |
| **Oracle Tier 2, L=inf, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 4** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.12% |
| **Oracle Tier 2, L=inf, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 5** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Tier 2, L=inf, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 0** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=inf, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 1** | 97.9 ± 1.4% | 95.5 ± 0.0% | 98.9 ± 0.5% | 78.3 ± 0.0% | 19.1 ± 0.0% | 32.79 ± 2.24% |
| **Oracle Tier 2, L=inf, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 2** | 61.3 ± 1.1% | 94.8 ± 0.0% | 99.0 ± 0.7% | 78.3 ± 0.0% | 0.0 ± 0.0% | 33.33 ± 2.61% |
| **Oracle Tier 2, L=inf, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 3** | 77.9 ± 1.4% | 89.6 ± 0.0% | 98.8 ± 0.7% | 82.7 ± 0.0% | 0.2 ± 0.0% | 29.85 ± 2.35% |
| **Oracle Tier 2, L=inf, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 4** | 97.4 ± 1.7% | 94.8 ± 0.0% | 99.1 ± 0.4% | 82.7 ± 0.0% | 19.1 ± 0.0% | 29.73 ± 2.02% |
| **Oracle Tier 2, L=inf, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 5** | 54.2 ± 1.2% | 95.6 ± 0.0% | 98.9 ± 0.7% | 87.1 ± 0.0% | 36.7 ± 0.0% | 34.58 ± 2.62% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Tier 2, L=50, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 0** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=50, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 1** | 0.0 ± 0.0% | 15.8 ± 0.0% | 19.0 ± 10.4% | 66.5 ± 0.0% | 100.0 ± 0.0% | 0.11 ± 0.13% |
| **Oracle Tier 2, L=50, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 2** | 0.0 ± 0.0% | 30.4 ± 0.0% | 24.2 ± 13.2% | 72.8 ± 0.0% | 100.0 ± 0.0% | 0.12 ± 0.13% |
| **Oracle Tier 2, L=50, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 3** | 0.0 ± 0.0% | 33.4 ± 0.0% | 28.7 ± 15.5% | 72.6 ± 0.0% | 100.0 ± 0.0% | 0.12 ± 0.13% |
| **Oracle Tier 2, L=50, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 4** | 0.0 ± 0.0% | 46.9 ± 0.0% | 37.8 ± 18.6% | 72.8 ± 0.0% | 100.0 ± 0.0% | 0.12 ± 0.13% |
| **Oracle Tier 2, L=50, With Rehearsal** | Oracle | Gate ON (Rollback) | **Cycle 5** | 0.0 ± 0.0% | 54.6 ± 0.0% | 43.1 ± 20.2% | 75.1 ± 0.0% | 100.0 ± 0.0% | 0.13 ± 0.14% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Tier 2, L=50, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 0** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=50, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 1** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=50, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 2** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=50, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 3** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.12% |
| **Oracle Tier 2, L=50, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 4** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.12% |
| **Oracle Tier 2, L=50, No Rehearsal (Rollback)** | Oracle | Gate ON (Rollback) | **Cycle 5** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Tier 2, L=50, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 0** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Oracle Tier 2, L=50, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 1** | 99.0 ± 0.7% | 98.1 ± 0.0% | 99.8 ± 0.1% | 94.7 ± 0.0% | 98.1 ± 0.0% | 87.49 ± 1.58% |
| **Oracle Tier 2, L=50, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 2** | 98.4 ± 1.0% | 97.2 ± 0.0% | 98.6 ± 1.9% | 83.4 ± 0.0% | 97.3 ± 0.0% | 74.21 ± 2.60% |
| **Oracle Tier 2, L=50, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 3** | 97.4 ± 1.7% | 95.3 ± 0.0% | 86.6 ± 4.4% | 80.0 ± 0.0% | 95.5 ± 0.0% | 72.47 ± 3.70% |
| **Oracle Tier 2, L=50, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 4** | 99.5 ± 0.3% | 99.1 ± 0.0% | 100.0 ± 0.0% | 94.8 ± 0.0% | 98.2 ± 0.0% | 94.22 ± 1.08% |
| **Oracle Tier 2, L=50, No Rehearsal (Gate OFF: True Forgetting)** | Oracle | Gate OFF (True Forgetting) | **Cycle 5** | 99.0 ± 0.7% | 98.1 ± 0.0% | 99.8 ± 0.1% | 90.4 ± 0.0% | 98.2 ± 0.0% | 83.19 ± 2.27% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trained Tier 2 (Pseudo-Labels), L=inf, With Rehearsal** | Trained Tier 2 (Pseudo-Labels) | Gate ON (Rollback) | **Cycle 0** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 2.9 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Trained Tier 2 (Pseudo-Labels), L=inf, With Rehearsal** | Trained Tier 2 (Pseudo-Labels) | Gate ON (Rollback) | **Cycle 1** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 1.2 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Trained Tier 2 (Pseudo-Labels), L=inf, With Rehearsal** | Trained Tier 2 (Pseudo-Labels) | Gate ON (Rollback) | **Cycle 2** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.7 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Trained Tier 2 (Pseudo-Labels), L=inf, With Rehearsal** | Trained Tier 2 (Pseudo-Labels) | Gate ON (Rollback) | **Cycle 3** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Trained Tier 2 (Pseudo-Labels), L=inf, With Rehearsal** | Trained Tier 2 (Pseudo-Labels) | Gate ON (Rollback) | **Cycle 4** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |
| **Trained Tier 2 (Pseudo-Labels), L=inf, With Rehearsal** | Trained Tier 2 (Pseudo-Labels) | Gate ON (Rollback) | **Cycle 5** | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.09 ± 0.13% |

> **Arms Description & Findings:**  
> 1. **Oracle Tier 2 vs Trained Tier 2 Pseudo-Labels:** Oracle runs represent the theoretical upper bound where escalated flows receive true ground-truth labels. Under realistic Trained Tier 2 pseudo-labels ($RF_{76}$ trained on baseline), Tier 2 cannot correctly label novel keep-alives and slowloris flows, demonstrating why human-in-the-loop verification or active learning is essential.  
> 2. **No-Rehearsal (Rollback vs Gate OFF):** Under No-Rehearsal with Gate ON, regression on historical FTP traffic triggers the safety gate, causing automatic rollback to baseline $v1.0$ (FTP recall stays 100%). Under No-Rehearsal with Gate OFF (unconditional deployment), the model suffers true catastrophic forgetting: FTP recall plummets to 36.7% and benign FPR surges to 34.6%.

---

## 5. Feedback Poisoning: Unified 25-Seed Evaluation

Conducted across 25 random seeds (`seeds = 100..124`) using CSE-CIC-IDS2018 (`lofo_FTP-BruteForce_train` partition: $N=8,625$ flows; 446 validation attack flows, 446 test attack flows). Injected flows are true attack flows labeled as benign.

- **Target Recall ($R_{\text{target}}$):** Percentage of test attack flows correctly identified as malicious by candidate model.
- **Attack Success ($S_{\text{attack}}$):** 100% - Target Recall (percentage of test attacks evading detection as false negatives).
- **Rejection Policy Note:** In single-cycle adaptation, rejections are 0% below 3.0% rate and peak at ~16% at 5.0%; in 5-cycle adaptation, average cycle rejections reach ~65% at 5%-10% rate. Rejection is NOT unconditionally 100%.

### 5.1 Single-Cycle Poisoning Sweep (25 Seeds, Clean Row Included)

| Injection Rate (%) | Gate Rejection Rate (Gate ON) (%) | $\Delta$ Val Acc (%) | $\Delta$ Historical Recall (%) | Candidate Target Recall (%) | Attack Success Without Cap (Gate OFF) (%) | Attack Success With Cap (Gate OFF) (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0%** | 0.0 ± 0.0% | 0.00 ± 0.00% | 0.00 ± 0.00% | 91.11 ± 0.54% | 8.89 ± 0.54% | **8.89 ± 0.54%** |
| **0.5%** | 0.0 ± 0.0% | 0.08 ± 0.05% | 0.16 ± 0.10% | 91.01 ± 0.54% | 8.99 ± 0.54% | **8.88 ± 0.55%** |
| **1.0%** | 0.0 ± 0.0% | 0.09 ± 0.05% | 0.25 ± 0.12% | 90.63 ± 0.61% | 9.37 ± 0.61% | **9.21 ± 0.59%** |
| **1.5%** | 0.0 ± 0.0% | 0.09 ± 0.06% | 0.20 ± 0.14% | 90.53 ± 0.57% | 9.47 ± 0.57% | **9.14 ± 0.51%** |
| **2.0%** | 0.0 ± 0.0% | 0.10 ± 0.05% | 0.39 ± 0.17% | 90.22 ± 0.61% | 9.78 ± 0.61% | **9.07 ± 0.60%** |
| **3.0%** | 4.0 ± 6.0% | 0.18 ± 0.08% | 0.59 ± 0.25% | 90.01 ± 0.61% | 9.99 ± 0.61% | **9.18 ± 0.61%** |
| **5.0%** | 16.0 ± 14.0% | 0.24 ± 0.10% | 1.00 ± 0.33% | 89.02 ± 0.65% | 10.98 ± 0.65% | **9.66 ± 0.61%** |
| **7.5%** | 12.0 ± 14.0% | 0.25 ± 0.09% | 1.01 ± 0.27% | 89.00 ± 0.63% | 11.00 ± 0.63% | **9.50 ± 0.56%** |
| **10.0%** | 12.0 ± 14.0% | 0.27 ± 0.10% | 1.01 ± 0.31% | 89.19 ± 0.64% | 10.81 ± 0.64% | **9.60 ± 0.55%** |

### 5.2 Multi-Cycle (5 Consecutive Cycles) Poisoning Sweep (25 Seeds)

| Injection Rate (%) | Rejections Without Cap (Gate ON) (%) | Target Recall Without Cap (Gate ON) (%) | Attack Success Without Cap (Gate ON) (%) | Rejections With Cap (Gate ON) (%) | Target Recall With Cap (Gate ON) (%) | Attack Success With Cap (Gate ON) (%) |
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
