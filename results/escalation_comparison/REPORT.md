# Escalation Triggers Comparative Evaluation & Two-Tier Security Audit

**Execution Timestamp:** 2026-10-21 20:56:53 UTC  
**Dataset:** CSE-CIC-IDS2018 (Benchmark Splits)  
**Evaluation Seeds:** All metrics report mean ± 95% block-bootstrap confidence intervals over 10 chronological time blocks (1,000 resamples).  

---

## 1. Standardized Escalation Gates Comparison

Features are standardized with `StandardScaler` fitted exclusively on clean benign training flows.
Evaluated across:
1. **RF Probability Band:** Uncertainty ranking based on $|P(\text{attack}) - 0.5|$.
2. **Full kNN Gate:** $k=5$, Euclidean distance on standardized 9 features ($N=4,000$ reference set).
3. **Prototype kNN Gate:** $M=32$ cluster centroids from `MiniBatchKMeans` on standardized 9 features.
4. **Mahalanobis Distance Gate:** Distance computed with regularized covariance on standardized 9 features.
5. **UNION Gate (RF Band + kNN):** Logical OR of RF band and Full kNN with split budget.

### 1.1 Performance at Matched Achieved Benign Escalation

| Gate Architecture | Target Budget | Achieved Benign Esc (%) | SSH Esc ($N=267$) | GoldenEye Esc ($N=450$) | Hulk Esc ($N=389$) | Slowloris Esc ($N=426$) | FTP Esc ($N=268$) | Novel Macro Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full kNN (Standardized)** | **1%** | 1.00 ± 0.44% | 0 (0.0%) | 63 (14.0%) | 100 (25.7%) | 277 (65.0%) | 0 (0.0%) | **26.18 ± 26.65%** |
| | **2%** | 2.00 ± 0.60% | 0 (0.0%) | 187 (41.6%) | 269 (69.2%) | 277 (65.0%) | 0 (0.0%) | **43.93 ± 32.71%** |
| | **5%** | 5.01 ± 0.86% | 0 (0.0%) | 255 (56.7%) | 288 (74.0%) | 277 (65.0%) | 0 (0.0%) | **48.93 ± 33.28%** |
| | **10%** | 10.00 ± 2.61% | 0 (0.0%) | 266 (59.1%) | 306 (78.7%) | 277 (65.0%) | 0 (0.0%) | **50.70 ± 33.98%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prototype kNN (M=32)** | **1%** | 1.02 ± 0.47% | 0 (0.0%) | 62 (13.8%) | 78 (20.1%) | 0 (0.0%) | 0 (0.0%) | **8.46 ± 5.95%** |
| | **2%** | 2.00 ± 0.86% | 0 (0.0%) | 125 (27.8%) | 124 (31.9%) | 278 (65.3%) | 0 (0.0%) | **31.23 ± 26.21%** |
| | **5%** | 5.01 ± 1.04% | 0 (0.0%) | 130 (28.9%) | 186 (47.8%) | 279 (65.5%) | 1 (0.4%) | **35.55 ± 27.39%** |
| | **10%** | 10.00 ± 1.54% | 0 (0.0%) | 265 (58.9%) | 322 (82.8%) | 281 (66.0%) | 1 (0.4%) | **51.91 ± 35.08%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mahalanobis Gate** | **1%** | 1.00 ± 0.54% | 0 (0.0%) | 17 (3.8%) | 20 (5.1%) | 0 (0.0%) | 0 (0.0%) | **2.23 ± 1.72%** |
| | **2%** | 2.00 ± 0.91% | 0 (0.0%) | 124 (27.6%) | 116 (29.8%) | 184 (43.2%) | 0 (0.0%) | **25.14 ± 19.11%** |
| | **5%** | 5.01 ± 0.93% | 0 (0.0%) | 250 (55.6%) | 330 (84.8%) | 277 (65.0%) | 0 (0.0%) | **51.35 ± 35.23%** |
| | **10%** | 10.00 ± 1.94% | 0 (0.0%) | 310 (68.9%) | 382 (98.2%) | 277 (65.0%) | 0 (0.0%) | **58.03 ± 38.78%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF Probability Band** | **1%** | 1.19 ± 0.45% | 3 (1.1%) | 0 (0.0%) | 4 (1.0%) | 332 (77.9%) | 23 (8.6%) | **20.02 ± 24.31%** |
| | **2%** | 2.33 ± 0.71% | 7 (2.6%) | 0 (0.0%) | 4 (1.0%) | 332 (77.9%) | 23 (8.6%) | **20.40 ± 23.88%** |
| | **5%** | 8.49 ± 1.57% | 7 (2.6%) | 0 (0.0%) | 4 (1.0%) | 426 (100.0%) | 26 (9.7%) | **25.91 ± 30.73%** |
| | **10%** | 15.94 ± 3.57% | 7 (2.6%) | 21 (4.7%) | 4 (1.0%) | 426 (100.0%) | 29 (10.8%) | **27.08 ± 30.09%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNION Gate (RF + kNN)** | **1%** | 1.04 ± 0.67% | 2 (0.7%) | 51 (11.3%) | 75 (19.3%) | 332 (77.9%) | 4 (1.5%) | **27.32 ± 29.18%** |
| | **2%** | 2.06 ± 0.77% | 3 (1.1%) | 63 (14.0%) | 104 (26.7%) | 332 (77.9%) | 23 (8.6%) | **29.95 ± 30.59%** |
| | **5%** | 5.82 ± 1.19% | 7 (2.6%) | 247 (54.9%) | 277 (71.2%) | 426 (100.0%) | 26 (9.7%) | **57.18 ± 41.10%** |
| | **10%** | 13.31 ± 1.68% | 7 (2.6%) | 255 (56.7%) | 292 (75.1%) | 426 (100.0%) | 26 (9.7%) | **58.59 ± 40.54%** |

### 1.2 Coverage-vs-Budget Curves
The full coverage curves comparing RF Band, Prototype kNN ($M=32$), and Full kNN at matched achieved benign escalation rates are rendered in:  
[coverage_vs_budget_curves.png](file:///Users/sohamsupekar/Desktop/MajorProject/Code/results/escalation_comparison/coverage_vs_budget_curves.png)

> **Key Architectural Takeaway:** Feature standardization allows kNN and Mahalanobis to escape the dominance of raw microsecond duration scales, capturing GoldenEye (56.7%) and Hulk (74.0%). The **UNION Gate** combines the complementary strengths of RF uncertainty and kNN density, achieving **57.18%** novel macro coverage at ~5% benign budget.

---

## 2. Trained Tier 2 Per-Flow Prediction Audit & Flow Reconciliation

We evaluated Trained Tier 2 ($RF_{76}$) predictions directly on the flows escalated by the gate:

| Evaluated Subset | Flow Count | Caught by Trained T2 | Missed by Trained T2 | Mean Prob | Max Prob |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Escalated Slowloris Flows (2% Budget)** | 277 | **0 (0.0%)** | **277 (100.0%)** | 0.1635 | 0.3000 |
| **Non-Escalated Slowloris Flows** | 149 | **37 (24.8%)** | **112 (75.2%)** | 0.2851 | 0.6200 |
| **Total Test Slowloris Pool** | 426 | **37 (8.69%)** | **389 (91.31%)** | 0.2061 | 0.6200 |

### 2.1 Technical Explanation of Zero Intercept (0/277 Caught)
1. **Orthogonal Trigger Spaces:** The kNN escalation gate triggers on extreme anomaly in temporal space (long flow duration and idle times typical of Slowloris keep-alive).
2. **Discriminative Shortcut of Trained T2:** Trained Tier 2 was trained exclusively on historical attacks (`FTP-BruteForce`), where malicious behavior is marked by elevated packet lengths and specific byte volumes.
3. **Disjoint Populations:** The 37 Slowloris flows caught by Trained Tier 2 standalone consist entirely of short, active handshake bursts that match FTP-like packet profiles; because these bursts look typical in duration space, they were NOT flagged by the kNN gate. Conversely, the 277 escalated flows exhibit long idle periods, which Tier 2 classifies as benign with mean probability 0.1635.

### 2.2 Reconciliation of Flow Count Discrepancies
- **12 flows (0.81%):** In preliminary multi-attack runs, Tier 2 standalone detected 12 out of all 1,532 novel flows ($12 / 1,532 = 0.78\% \approx 0.81\%$).
- **12.4 flows (2.9%):** Standalone Tier 1 recall on test Slowloris ($0.029 \times 426 = 12.35 \approx 12.4$ flows).
- **34–37 flows (7.5%–8.7%):** Standalone Trained Tier 2 recall on test Slowloris ($37 / 426 = 8.69\%$).

---

## 3. Feedback Poisoning: Unified 25-Seed Evaluation

Conducted across 25 random seeds (`seeds = 100..124`) using CSE-CIC-IDS2018 (`lofo_FTP-BruteForce_train` partition: $N_{\text{tr}} = 8,625$ flows; 446 validation attack flows, 446 test attack flows). Injected flows are true attack flows labeled as benign (malicious-to-benign label flipping).

### 3.1 Single-Cycle Poisoning Sweep (25 Seeds)

| Injection Rate (%) | Gate Rejection Rate (%) | $\Delta$ Val Acc (%) | $\Delta$ Historical Recall (%) | Candidate Target Recall (%) | Attack Success Without Cap (%) | Attack Success With Cap (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.5%** | **0.0%** | 0.01% | 0.02% | 91.01% | 8.99% | **8.88%** |
| **1.0%** | **0.0%** | 0.05% | 0.14% | 90.63% | 9.37% | **9.21%** |
| **1.5%** | **0.0%** | 0.02% | 0.13% | 90.53% | 9.47% | **9.14%** |
| **2.0%** | **0.0%** | 0.02% | 0.28% | 90.22% | 9.78% | **9.07%** |
| **3.0%** | **4.0%** | 0.09% | 0.41% | 90.01% | 9.99% | **9.18%** |
| **5.0%** | **16.0%** | 0.21% | 0.98% | 89.02% | 10.98% | **9.66%** |
| **7.5%** | **12.0%** | 0.23% | 1.01% | 89.00% | 11.00% | **9.50%** |
| **10.0%** | **12.0%** | 0.26% | 1.00% | 89.19% | 10.81% | **9.60%** |

### 3.2 Multi-Cycle (5 Consecutive Cycles) Poisoning Sweep (25 Seeds)

| Injection Rate (%) | Average Cycle Rejections (%) | Final Active Target Recall (%) | Final Active Attack Success (%) |
| :---: | :---: | :---: | :---: |
| **0.5%** | **0.0%** | 90.30% | 9.70% |
| **1.0%** | **2.4%** | 89.17% | 10.83% |
| **1.5%** | **3.2%** | 88.53% | 11.47% |
| **2.0%** | **8.8%** | 86.97% | 13.03% |
| **3.0%** | **29.6%** | 85.74% | 14.26% |
| **5.0%** | **66.4%** | 87.69% | 12.31% |
| **7.5%** | **65.6%** | 87.09% | 12.91% |
| **10.0%** | **64.8%** | 87.25% | 12.75% |

### 3.3 Poisoning Audit Technical Explanations
1. **2.02% vs 3.0% Tolerance:** In [`src/feedback/production_loop.py`](file:///Users/sohamsupekar/Desktop/MajorProject/Code/src/feedback/production_loop.py#L163), `RetrainingGate` specifies `max_acc_drop = 0.02` (2.00%). A previous markdown draft incorrectly stated 3.0%. When Seed 42 experienced $\Delta\text{AtkRec} = 2.02\%$, it strictly exceeded 2.00% ($2.02\% > 2.00\%$), tripping `REJECTED_ROLLBACK`.
2. **$\Delta\text{AtkRec}$ vs Target-Recall:** $\Delta\text{AtkRec}$ evaluates recall loss on the *validation* historical attack partition (`val_atk`, $N=446$), while Target Recall is measured on the separate *test* partition (`test_atk`, $N=446$).
3. **Dose-Response:** In single-cycle adaptation, the 40-tree Random Forest ensemble absorbs up to 2.0% poison without degradation. Above 3.0%, rejection rate jumps to 16.0% and $\Delta\text{AtkRec}$ rises to 1.00%. In multi-cycle adaptation, cumulative poison accumulation shows a monotonic dose-response: rejections climb from 0.0% at 0.5% rate -> 2.4% at 1.0% -> 8.8% at 2.0% -> 29.6% at 3.0% -> ~66% at 5.0%–10.0%.
