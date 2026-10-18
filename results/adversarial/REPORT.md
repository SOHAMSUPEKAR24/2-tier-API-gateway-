# Phase 7 Report: Adversarial Robustness & Attack Evaluation (Hardened)

**Date:** 2026-10-18 13:10:52 UTC  
**Seeds Tested:** `[42, 71, 137, 256, 512]` (mean ± 95% CI reported)  

## 1. Band-Evasion Attack & Defenses

Attacks target ONLY test samples that Tier 1 correctly caught or escalated without perturbation. At $\epsilon=0$, evasion success rate is strictly 0.00%.

| Perturbation Budget (eps) | No Defense Evasion (%) | Randomized Thresh (%) | Adv Retraining (%) | Clustering Monitor (%) |
| :--- | :--- | :--- | :--- | :--- |
| 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 3.93 ± 2.64 | 0.00 ± 0.00 |
| 0.05 | 85.48 ± 40.32 | 57.39 ± 38.62 | 67.59 ± 27.81 | 26.72 ± 31.39 |
| 0.10 | 80.82 ± 53.25 | 51.70 ± 38.07 | 33.45 ± 51.59 | 7.46 ± 9.89 |
| 0.15 | 80.68 ± 53.63 | 59.61 ± 46.03 | 18.49 ± 51.35 | 39.23 ± 65.00 |
| 0.20 | 80.68 ± 53.63 | 54.43 ± 47.37 | 0.00 ± 0.00 | 39.23 ± 65.00 |
| 0.30 | 80.00 ± 55.53 | 61.27 ± 49.62 | 0.00 ± 0.00 | 37.69 ± 60.26 |
| 0.40 | 80.68 ± 53.63 | 61.66 ± 49.34 | 0.00 ± 0.00 | 37.63 ± 62.44 |
| 0.50 | 80.68 ± 53.63 | 60.64 ± 53.23 | 0.00 ± 0.00 | 37.63 ± 62.44 |

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
| 5% | 96.70% | 90.87 ± 2.09% | 23.08 ± 5.26% | 96.11 ± 0.56% | 9.01 ± 1.51% |
| 10% | 96.70% | 80.05 ± 0.82% | 50.86 ± 1.95% | 96.00 ± 0.47% | 9.36 ± 1.41% |
| 20% | 96.70% | 78.99 ± 2.98% | 53.53 ± 7.64% | 96.12 ± 0.42% | 9.11 ± 1.19% |
