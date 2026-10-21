# Phase 7 Report: Adversarial Robustness & Attack Evaluation (Hardened)

**Date:** 2026-10-21 20:56:49 UTC  
**Seeds Tested:** `[42, 71, 137, 256, 512]` (mean ± 95% CI reported)  

## 1. Band-Evasion Attack & Defenses

Attacks target ONLY test samples that Tier 1 correctly caught or escalated without perturbation. At $\epsilon=0$, evasion success rate is strictly 0.00%.

| Perturbation Budget (eps) | No Defense Evasion (%) | Randomized Thresh (%) | Adv Retraining (%) | Clustering Monitor (%) |
| :--- | :--- | :--- | :--- | :--- |
| 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 3.50 ± 6.39 | 0.00 ± 0.00 |
| 0.05 | 30.33 ± 13.22 | 21.33 ± 10.63 | 15.52 ± 11.85 | 16.29 ± 12.39 |
| 0.10 | 37.00 ± 1.11 | 26.04 ± 9.67 | 14.07 ± 13.30 | 19.81 ± 11.80 |
| 0.15 | 46.65 ± 7.85 | 35.53 ± 5.79 | 3.73 ± 10.02 | 24.46 ± 13.05 |
| 0.20 | 52.19 ± 4.09 | 41.88 ± 6.05 | 0.00 ± 0.00 | 33.79 ± 12.08 |
| 0.30 | 63.27 ± 4.15 | 53.30 ± 5.67 | 0.00 ± 0.00 | 47.98 ± 5.40 |
| 0.40 | 72.84 ± 3.71 | 66.92 ± 3.04 | 0.00 ± 0.00 | 62.76 ± 4.30 |
| 0.50 | 75.26 ± 4.03 | 62.87 ± 6.59 | 0.00 ± 0.00 | 53.31 ± 5.31 |

## 2. Escalation-Flood Attack Resilience

Simulates Tier 2 finite capacity ($\mu=200$ req/s) and bounded queue ($Q_{\max}=100$). Sweeps arrival rates above and below capacity.

| Flood Arrival Rate (req/s) | Baseline Max Q | Baseline Overflows | Baseline Benign p99 Latency (ms) | Controller Max Q | Controller Overflows | Controller Benign p99 Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50 | 0 | 0 | 25.0 ms | 0 | 0 | 25.0 ms |
| 100 | 0 | 0 | 45.0 ms | 0 | 0 | 45.0 ms |
| 150 | 0 | 0 | 65.0 ms | 0 | 0 | 65.0 ms |
| 200 | 0 | 0 | 90.0 ms | 0 | 0 | 90.0 ms |
| 250 | 50 | 0 | 352.6 ms | 50 | 0 | 352.6 ms |
| 300 | 80 | 170 | 505.0 ms | 79 | 0 | 494.6 ms |
| 400 | 80 | 620 | 505.0 ms | 80 | 16 | 505.0 ms |
| 500 | 80 | 1020 | 505.0 ms | 80 | 31 | 505.0 ms |

## 3. Feedback Poisoning Damage & Retraining Gate Protection

Evaluates Tier 1 trained on known families with high baseline accuracy (95.89%). Injects label-flipped pseudo-labels across 5 adaptation cycles.

| Injection Rate | Baseline Acc (%) | Naive Cycle 5 Acc (%) | Naive Atk Success (%) | Protected Cycle 5 Acc (%) | Protected Atk Success (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 5% | 97.20% | 82.84 ± 2.90% | 55.20 ± 9.16% | 96.62 ± 0.50% | 9.78 ± 1.81% |
| 10% | 97.20% | 80.15 ± 1.85% | 63.95 ± 5.89% | 96.63 ± 0.59% | 10.04 ± 1.77% |
| 20% | 97.20% | 82.85 ± 1.03% | 55.25 ± 3.37% | 96.69 ± 0.50% | 9.78 ± 1.79% |
