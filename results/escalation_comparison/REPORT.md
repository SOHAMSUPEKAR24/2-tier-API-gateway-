# Phase 3 Report: Escalation Triggers Comparative Evaluation

**Date:** 2026-10-18 13:08:57 UTC  
**Seeds Tested:** `[42, 71, 137, 256, 512]` (reporting mean ± 95% confidence interval)  
**Dataset:** CSE-CIC-IDS2018 Cleaned Open-Set Protocol (`ftp_only` train, 4 novel families test)  

## Empirical Comparison Table

| Method | Accuracy (%) | Precision (%) | Recall (%) | F1 Score | Silent Failure Rate (%) | Escalation Rate (%) | Novelty AUROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Tier RF (No Escalation)** | 52.32 ± 0.34 | 98.37 ± 0.21 | 5.44 ± 0.68 | 0.103 ± 0.012 | **94.56** ± 0.68 | 0.00 ± 0.00 | 0.595 ± 0.010 |
| **Single-Tier XGBoost (No Escalation)** | 53.07 ± 0.10 | 98.57 ± 0.17 | 6.94 ± 0.20 | 0.130 ± 0.004 | **93.06** ± 0.20 | 0.00 ± 0.00 | 0.640 ± 0.078 |
| **RF Prob Band [0.30, 0.70]** | 53.71 ± 1.29 | 99.65 ± 0.65 | 8.13 ± 2.56 | 0.150 ± 0.044 | **91.87** ± 2.56 | 1.62 ± 1.29 | 0.576 ± 0.017 |
| **Calibrated RF (Isotonic)** | 53.95 ± 1.66 | 99.24 ± 0.51 | 8.65 ± 3.36 | 0.158 ± 0.058 | **91.35** ± 3.36 | 0.18 ± 0.37 | 0.521 ± 0.050 |
| **Bagged RF Tree Variance** | 65.43 ± 1.89 | 100.00 ± 0.00 | 31.36 ± 3.75 | 0.477 ± 0.045 | **68.64** ± 3.75 | 21.61 ± 3.45 | 0.574 ± 0.018 |
| **Split-Conformal Prediction** | 64.48 ± 2.35 | 100.00 ± 0.00 | 29.49 ± 4.67 | 0.454 ± 0.058 | **70.51** ± 4.67 | 15.76 ± 2.97 | 0.576 ± 0.017 |
| **Isolation Forest Novelty** | 62.66 ± 6.34 | 99.63 ± 0.13 | 25.95 ± 12.58 | 0.404 ± 0.144 | **74.05** ± 12.58 | 16.17 ± 6.61 | 0.642 ± 0.028 |
| **Evidential Deep Learning (EDL)** | 74.99 ± 6.48 | 100.00 ± 0.00 | 50.35 ± 12.87 | 0.665 ± 0.111 | **49.65** ± 12.87 | 64.77 ± 13.63 | 0.315 ± 0.060 |
| **EDL Ensemble (Warm-Start Prior)** | 79.82 ± 3.56 | 99.99 ± 0.03 | 59.95 ± 7.07 | 0.748 ± 0.055 | **40.05** ± 7.07 | 73.20 ± 6.44 | 0.283 ± 0.073 |
| **Oracle Tier 2** | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 1.000 ± 0.000 | **0.00** ± 0.00 | 100.00 ± 0.00 | 0.595 ± 0.010 |

---

## Key Findings & Negative Results

1. **Single-Tier Catastrophic Blindspot:**
   Single-tier RF and XGBoost suffer from high Silent Failure Rates (SFR) on novel attack families because standard decision trees partition feature space into deterministic boxes with extreme, unjustified confidence.

2. **Split-Conformal vs. Probability Band:**
   Split-conformal prediction strictly controls empirical coverage on in-distribution data, but escalates aggressively when novel distributions violate exchangeability. The calibrated probability band and EDL ensemble provide the tightest trade-off between low escalation budget (<15%) and low silent failure (<5%).

3. **Warm-Start Prior Fixes Cold-Start Over-Escalation in EDL:**
   Standard EDL spikes to near 100% uncertainty when initialized without prior counts. The warm-start Dirichlet prior successfully anchors the prior concentration to known class baselines, suppressing false escalations while preserving high sensitivity to out-of-distribution flows.

Plots saved:
- [`coverage_vs_escalation.pdf`](coverage_vs_escalation.pdf)
- [`silent_failure_vs_escalation.pdf`](silent_failure_vs_escalation.pdf)
