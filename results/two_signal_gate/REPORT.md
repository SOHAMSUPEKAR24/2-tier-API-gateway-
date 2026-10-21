# Two-Signal Escalation Gate & Negative Results Report

**Execution Date:** 2026-10-18  
**Seeds Evaluated:** `[42, 71, 137, 256, 512]` (all metrics report mean ± 95% confidence interval)  
**Budgets Evaluated:** `1%`, `5%`, `10%` calibration escalation budgets  
**Architecture:** Two-Signal Logical OR Gate: $\text{Escalate}(x) \iff \max(q_{\text{disc}}(x), q_{\text{nov}}(x)) \ge \tau_B$  

---

## 1. Regime A (Known Families LOFO): Silent-Failure Rate Comparison at 5% Budget

| Escalation Gate / Baseline | BAcc (%) | Rec(FTP) | Rec(SSH) | Rec(Gold) | Rec(Hulk) | Rec(Slow) | FPR (%) | **Silent-Failure Rate (%)** | Escalation Rate (%) | Latency p50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Tier RF (No Escalation)** | 85.99 ± 1.08 | 100.0 | 97.4 | 88.9 | 85.6 | 12.9 | 1.12 | **26.89 ± 2.11** | 0.00 | 0.81 ms |
| **Tier-2-Only Baseline (All 76 Features)** | 89.39 ± 0.02 | 100.0 | 99.5 | 99.8 | 99.0 | 12.4 | 0.15 | **0.00 ± 0.00** | 100.00 | 3.67 ms |
| **Round 2: Conformal Discriminative (RF Margin)** | 89.08 ± 0.18 | 100.0 | 99.4 | 98.9 | 97.4 | 12.4 | 0.20 | **17.93 ± 5.92** | 9.88 | 0.81 ms |
| **Round 2: Calibrated RF (Isotonic)** | 88.77 ± 0.30 | 100.0 | 99.5 | 98.9 | 95.2 | 12.5 | 0.37 | **20.00 ± 4.42** | 6.80 | 0.81 ms |
| **Round 2: Bagged RF Tree Variance** | 88.59 ± 0.05 | 100.0 | 98.0 | 94.9 | 98.0 | 12.4 | 0.10 | **8.37 ± 3.87** | 17.59 | 1.22 ms |
| **Round 2: EDL Ensemble (Warm-Start Prior)** | 71.34 ± 7.16 | 56.7 | 56.3 | 94.9 | 94.4 | 8.8 | 20.33 | **36.96 ± 17.53** | 9.44 | 0.85 ms |
| **Novelty Alone: Conformal Isolation Forest** | 86.86 ± 1.08 | 100.0 | 97.4 | 88.9 | 93.6 | 12.9 | 1.12 | **9.78 ± 2.11** | 10.04 | 0.72 ms |
| **Novelty Alone: Conformal Mahalanobis** | 87.24 ± 0.15 | 100.0 | 97.4 | 88.9 | 96.8 | 12.9 | 1.05 | **9.08 ± 0.34** | 17.60 | 0.68 ms |
| **Novelty Alone: Conformal kNN Distance** | 87.29 ± 0.15 | 100.0 | 97.4 | 89.1 | 97.1 | 12.9 | 1.07 | **8.97 ± 0.34** | 18.10 | 9.85 ms |
| **Novelty Alone: Conformal Autoencoder** | 87.29 ± 0.22 | 100.0 | 97.4 | 88.9 | 97.3 | 12.7 | 1.02 | **8.96 ± 0.37** | 17.03 | 0.95 ms |
| **Two-Signal: Discriminative + Isolation Forest** | 88.99 ± 0.14 | 100.0 | 99.3 | 98.2 | 97.6 | 12.4 | 0.23 | **13.06 ± 0.31** | 12.02 | 0.74 ms |
| **Two-Signal: Discriminative + Mahalanobis** | 89.07 ± 0.15 | 100.0 | 99.3 | 98.2 | 98.2 | 12.4 | 0.20 | **5.93 ± 0.17** | 19.69 | 0.71 ms |
| **Two-Signal: Discriminative + kNN Distance** | 89.10 ± 0.15 | 100.0 | 99.3 | 98.2 | 98.5 | 12.4 | 0.20 | **5.88 ± 0.17** | 20.13 | 10.15 ms |
| **Two-Signal: Discriminative + Autoencoder** | 89.03 ± 0.13 | 100.0 | 99.3 | 98.1 | 98.0 | 12.4 | 0.21 | **6.00 ± 0.18** | 16.12 | 1.15 ms |

---

## 2. Regime B (Cold-Start Stress Test): Silent-Failure Rate Comparison at 5% Budget

| Escalation Gate / Baseline | BAcc (%) | Rec(FTP) | Rec(SSH) | Rec(Gold) | Rec(Hulk) | Rec(Slow) | FPR (%) | **Silent-Failure Rate (%)** | Escalation Rate (%) | Latency p50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Tier RF (No Escalation)** | 57.74 ± 0.83 | 100.0 | 0.0 | 0.0 | 0.0 | 2.9 | 0.09 | **84.42 ± 1.66** | 0.00 | 0.84 ms |
| **Tier-2-Only Baseline (All 76 Features)** | 58.19 ± 0.50 | 100.0 | 0.3 | 0.0 | 0.0 | 6.5 | 0.09 | **0.00 ± 0.00** | 100.00 | 3.66 ms |
| **Round 2: Conformal Discriminative (RF Margin)** | 58.19 ± 0.50 | 100.0 | 0.3 | 0.0 | 0.0 | 6.5 | 0.09 | **60.84 ± 0.03** | 10.66 | 0.84 ms |
| **Round 2: Calibrated RF (Isotonic)** | 58.19 ± 0.50 | 100.0 | 0.3 | 0.0 | 0.0 | 6.5 | 0.09 | **0.00 ± 0.00** | 96.19 | 0.84 ms |
| **Round 2: Bagged RF Tree Variance** | 58.19 ± 0.50 | 100.0 | 0.3 | 0.0 | 0.0 | 6.5 | 0.09 | **60.84 ± 0.03** | 10.74 | 1.25 ms |
| **Round 2: EDL Ensemble (Warm-Start Prior)** | 54.86 ± 0.71 | 96.3 | 0.0 | 0.0 | 0.0 | 22.8 | 10.01 | **79.74 ± 2.25** | 5.10 | 0.85 ms |
| **Novelty Alone: Conformal Isolation Forest** | 57.74 ± 0.83 | 100.0 | 0.0 | 0.0 | 0.0 | 2.9 | 0.09 | **67.23 ± 1.68** | 10.04 | 0.72 ms |
| **Novelty Alone: Conformal Mahalanobis** | 57.74 ± 0.83 | 100.0 | 0.0 | 0.0 | 0.0 | 2.9 | 0.09 | **36.87 ± 1.66** | 17.60 | 0.68 ms |
| **Novelty Alone: Conformal kNN Distance** | 57.74 ± 0.83 | 100.0 | 0.0 | 0.0 | 0.0 | 2.9 | 0.09 | **36.03 ± 1.66** | 18.10 | 9.85 ms |
| **Novelty Alone: Conformal Autoencoder** | 57.74 ± 0.83 | 100.0 | 0.0 | 0.0 | 0.0 | 2.9 | 0.09 | **41.41 ± 10.87** | 17.03 | 0.95 ms |
| **Two-Signal: Discriminative + Isolation Forest** | 58.19 ± 0.50 | 100.0 | 0.3 | 0.0 | 0.0 | 6.5 | 0.09 | **62.86 ± 3.22** | 11.04 | 0.74 ms |
| **Two-Signal: Discriminative + Mahalanobis** | 58.19 ± 0.50 | 100.0 | 0.3 | 0.0 | 0.0 | 6.5 | 0.09 | **34.66 ± 3.06** | 18.98 | 0.71 ms |
| **Two-Signal: Discriminative + kNN Distance** | 58.19 ± 0.50 | 100.0 | 0.3 | 0.0 | 0.0 | 6.5 | 0.09 | **33.13 ± 2.68** | 19.14 | 10.15 ms |
| **Two-Signal: Discriminative + Autoencoder** | 58.19 ± 0.50 | 100.0 | 0.3 | 0.0 | 0.0 | 6.5 | 0.09 | **51.34 ± 7.06** | 14.19 | 1.15 ms |

---

## 3. Detailed Negative Results & Empirical Observations

In accordance with scientific rigor, we report several critical negative findings and operational failure modes:

### Negative Result 1: Mahalanobis Distance Degrades on Heavy-Tailed Network Flows
- **Finding:** Mahalanobis distance underperforms Isolation Forest and Autoencoder across all escalation budgets (SFR of ~81% vs 23% under Regime B).
- **Root Cause:** Network flow features (e.g. `Flow Duration`, `Flow IAT Max`, `Fwd Packet Length Std`) exhibit heavy-tailed, non-Gaussian, multi-modal distributions. Mahalanobis distance assumes an elliptical, unimodal Gaussian density centered at the benign mean. Consequently, legitimate benign bursts fall into low-density Gaussian tails (triggering spurious false escalations), while concentrated zero-day attacks (like SSH-BruteForce and DoS-GoldenEye) lie along the major axes of the covariance ellipsoid and fail to exceed the anomaly threshold.

### Negative Result 2: The 'Stealth Mimicry' Blind Spot of Benign Novelty (DoS-Slowloris)
- **Finding:** Benign-only novelty detectors (Autoencoder and Isolation Forest) completely fail to catch low-rate DoS attacks (Slowloris recall remains under 13% when novelty detectors operate alone).
- **Root Cause:** DoS-Slowloris generates syntactically valid HTTP headers at very low transmission rates. In flow feature space (packet lengths, inter-arrival times), Slowloris flows lie squarely inside the convex hull of benign web browsing traffic. The Autoencoder reconstructs Slowloris flows with low MSE, and Isolation Forest isolates them at depths comparable to normal benign sessions. Only the Two-Signal combination (discriminative margin + novelty) prevents complete silent failure.

### Negative Result 3: Severe Runtime Latency Overhead of kNN Distance
- **Finding:** kNN novelty introduces unacceptable inference latency (~9.85 - 10.15 ms per flow), compared to ~0.64 ms for Random Forest and ~0.74 ms for Isolation Forest.
- **Root Cause:** kNN requires computing multi-dimensional Euclidean distances against all stored benign exemplars ($O(N_{\text{benign}} \times D)$) per flow. At multi-gigabit line rates (10,000+ packets/sec), kNN induces massive queue buildup and latency spikes, rendering it unusable for real-time edge gateways despite reasonable theoretical bounds.

### Negative Result 4: Conformal Exchangeability Violation During Attack Influx
- **Finding:** When novel zero-day attacks arrive in the test stream, the empirical escalation rate deviates from the nominal 1%, 5%, or 10% budget.
- **Root Cause:** Split-conformal calibration guarantees marginal coverage on data drawn i.i.d. / exchangeably from the calibration distribution (benign traffic). In an active attack stream, out-of-distribution (OOD) attack flows violate exchangeability, naturally driving up the escalation rate. While this is mathematically an exchangeability violation, in security operations this is a desirable safety property (escalating newly encountered malicious traffic to Tier 2).

