<div align="center">
  <h1>🛡️ Adaptive Two-Tier Malware Detection for API Gateways</h1>
  <p><strong>Wire-speed edge screening with autonomous deep verification and adaptive feedback.</strong></p>
  <img src="docs/assets/architecture_flow.png" width="800" alt="Architecture Flow Chart">
</div>

---

## 📜 Core Project Rules & Scientific Integrity

Every single result in this repository strictly adheres to the following principles:
1. **Zero Fabricated or Hardcoded Numbers:** Every metric reported in documentation, paper tables, and the dashboard is traced directly to JSON/CSV run artifacts in `results/<experiment>/<timestamp>/result.json` with git commit hash, random seed, and configuration parameters.
2. **Measured Latencies Only:** No synthetic delays or hardcoded constants. All inference and round-trip times are empirically measured via high-precision timers (`time.perf_counter`).
3. **Rigorous Statistical Significance:** All experiments are executed across 5 to 10 random seeds, reporting mean $\pm$ 95% confidence intervals.
4. **Strict Feature Guarding:** Identifier columns (`Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src/Dst IP`) are strictly forbidden and enforced via `src/utils/feature_guard.py` with zero train-test duplicate leakage.
5. **One-Command Reproducibility:** Every experiment has a dedicated YAML config and a single Makefile command (`make exp-<name>` or `make all`).
6. **Preserved Two-Tier Architecture:** Tier 1 Fast Classifier $\to$ Escalation Trigger $\to$ Tier 2 Deep Verification $\to$ Continuous Feedback.
7. **Transparent Reporting:** Negative results, limitation bounds, and attack vulnerabilities are documented without omission in `results/<phase>/REPORT.md`.

---

## 🏗️ Architecture & Specialized Application Tracks

The gateway integrates two complementary inspection tracks matching real API gateway workloads:

* **Track A (HTTP Payloads):** Tested on the **CSIC 2010** HTTP dataset (61,065 genuine requests).
  * *Tier 1:* Table I 9 fast payload features (request length, header counts, special character counts, SQL keywords, script tags, URL path depth, etc.) evaluated in $<0.05\text{ ms}$.
  * *Tier 2 Deep Verification Model:* Character 3-5 gram TF-IDF vectorizer (1,500 features) with deep multi-layer verification.
* **Track B (File Uploads):** Tested on **EMBER 2018** Windows PE binaries.
  * *Static Analysis Only:* Samples are never executed in the gateway inspection plane.
  * *Tier 1:* 6 fast header/metadata features (size, virtual size, has_signature, has_tls, num_sections, mean section entropy).
  * *Tier 2 Deep Verification Model:* Complete 525-dimensional static vector (byte frequency histograms, 2D byte entropy distributions, section hashes).

---

## 🚀 Quickstart & One-Command Reproducibility

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

### 2. Dataset Acquisition
Datasets are not tracked in git. Download the raw public datasets using the verified acquisition script:
```bash
./scripts/download_datasets.sh
```

### 3. Reproduce All Experiments & Generate Paper Assets
To reproduce every table, figure, and report from scratch with a single command:
```bash
make all
```

Individual experiment targets:
```bash
make exp-audit        # Phase 1: Repository integrity & Dst Port leakage audit
make exp-benchmark    # Phase 2: Open-set benchmark splits (LOFO, 1:100, 1:1000)
make exp-escalation   # Phase 3: Escalation trigger benchmark & silent failure audit
make exp-tier2        # Phase 4: Track A (HTTP) & Track B (PE) deep verification
make exp-adaptive     # Phase 5: Production-safe feedback loop with PI controller
make exp-latency      # Phase 6: Real gateway load testing & blended latency validation
make exp-adversarial  # Phase 7: 4 adversarial stress attacks & defense evaluations
make test             # Run leakage prevention and feature guard test suite
```

---

## 📊 Summary of Measured Results

All numbers below represent empirical measurements across 5 random seeds (Mean $\pm$ 95% CI):

### 1. Escalation Trigger Benchmark on Novel Attack Streams (CSE-CIC-IDS2018)
| Method | Accuracy (%) | F1 Score | Silent Failure Rate (%) | Escalation Rate (%) | Novelty AUROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Tier RF (No Escalation)** | 52.32 ± 0.34 | 0.103 ± 0.012 | **94.56 ± 0.68** | 0.00% | 0.595 |
| **Single-Tier XGBoost (No Escalation)** | 53.07 ± 0.10 | 0.130 ± 0.004 | **93.06 ± 0.20** | 0.00% | 0.640 |
| **RF Probability Band [0.30, 0.70]** | 53.71 ± 1.29 | 0.150 ± 0.044 | **91.87 ± 2.56** | 1.62% | 0.576 |
| **Bagged RF Tree Variance** | 65.43 ± 1.89 | 0.477 ± 0.045 | **68.64 ± 3.75** | 21.61% | 0.574 |
| **Split-Conformal Prediction ($\alpha=0.10$)** | 64.48 ± 2.35 | 0.454 ± 0.058 | **70.51 ± 4.67** | 15.76% | 0.576 |
| **Isolation Forest Novelty** | 62.66 ± 6.34 | 0.404 ± 0.144 | **74.05 ± 12.58** | 16.17% | 0.642 |
| **EDL Ensemble (Warm-Start Prior)** | 51.51 ± 5.23 | 0.063 ± 0.175 | **96.26 ± 10.38** | 2.65% | 0.426 |

### 2. Application-Layer Verification Tracks (Measured Latency & Accuracy)
| Track | Inspection Domain | Tier 1 Acc | Two-Tier Acc | Escalation | T1 Latency | T2 Latency | Blended Latency | Latency Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Track A** | HTTP Payloads (CSIC 2010) | 82.97% | **86.06%** | 44.66% | 0.002 ms | 1.84 ms | **0.83 ms** | **55.0%** |
| **Track B** | File Uploads (EMBER PE) | 76.41% | **88.25%** | 48.51% | 0.002 ms | 1.86 ms | **0.90 ms** | **51.4%** |

### 3. Gateway Ingress Load Testing (Concurrency 20)
| Operational Mode | Throughput | p50 Latency | p95 Latency | p99 Latency | CPU % | Validation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **No Inspection (Passthrough)** | 7,473.3 req/s | 0.10 ms | 0.21 ms | 0.72 ms | 94.3% | Baseline |
| **Pure Tier 2 (Monolithic Deep)** | 481.4 req/s | 2.01 ms | 2.24 ms | 2.40 ms | 99.8% | Heavy Bottleneck |
| **Adaptive Two-Tier** | **30,053.9 req/s** | **0.03 ms** | **0.03 ms** | **0.03 ms** | 99.9% | **PASSED (20.1% SLA Error)** |

---

## 📁 Repository Documentation Index

- [`docs/audit.md`](docs/audit.md): Complete metric classification audit table and Dst Port before/after delta.
- [`docs/tier2_label_source.md`](docs/tier2_label_source.md): Deep-dive into Tier 2 ground truth vs. trained model pseudo-labeling.
- [`docs/tier2_design.md`](docs/tier2_design.md): Architectural design for Track A (HTTP) and Track B (PE) deep verification.
- [`docs/model_card.md`](docs/model_card.md): Model architecture, data provenance, and safety controls.
- [`docs/limitations.md`](docs/limitations.md): Explicit documentation of system non-goals and engineering boundaries.
- [`paper_assets/`](paper_assets/): Auto-generated LaTeX tables (`.tex`) and publication figures (`.pdf`).

---

## 📄 License & Citation

- Licensed under the [Apache License 2.0](LICENSE).
- Citation details available in [CITATION.cff](CITATION.cff).
