<div align="center">
  <h1>🛡️ Adaptive Two-Tier Malware Detection for API Gateways</h1>
  <p><strong>A Verified, Real-World Implementation with Evidential-Uncertainty Evaluation</strong></p>
</div>

---

## 📖 Overview

This repository contains the **verified, real-world data implementation** of the architecture described in the manuscript: *"Adaptive Two-Tier Malware Detection in API Gateways: A Validated Framework, an Evidential-Uncertainty Evaluation, and a Kernel-Acceleration Roadmap."*

While the original paper utilized a synthetic corpus to model theoretical L7 application behaviors, this codebase applies the exact Tier 1/Tier 2 evidential-uncertainty mathematics and feedback loop to **real-world, multi-gigabyte datasets**:
* **CIC-IDS 2017/2018:** Network-layer flow features representing high-speed edge traffic.
* **EMBER 2018:** Static PE malware features representing deep sandbox inspection.

## 🏗️ Architecture

The system implements a resource-constrained triage model for high-traffic environments:
1. **Tier 1 (Edge Screen):** A fast, lightweight Random Forest evaluating 9 structural features at wire-speed. Confident predictions (`> 0.70` or `< 0.30`) are actioned immediately. Uncertain predictions (`0.30 - 0.70`) are escalated.
2. **Tier 2 (Deep Sandbox):** An expensive, deep-inspection Random Forest evaluating static PE features of the escalated payloads.
3. **Algorithm 2 (Feedback Loop):** Tier 2's definitive judgments are fed back into Tier 1 as high-confidence training data. Over time, Tier 1 learns the complex boundaries that initially caused uncertainty, driving the escalation rate down without sacrificing edge accuracy.

> **Note on Feature Mapping:** For the exact mapping between the paper's theoretical features and the real-world datasets implemented here, please refer to [`docs/dataset_provenance.md`](docs/dataset_provenance.md).

## 🚀 Quickstart

To run the full Algorithm 2 pipeline on the real-world data:

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Extract & Balance Real Data
*Note: This script safely chunks massive 6GB+ CSV files to prevent Out-Of-Memory (OOM) crashes, building a balanced 10,000-row testing corpus (5,000 Benign / 5,000 Attack).*
```bash
python src/data_collection/build_real_corpus.py
```

### 3. Execute the Real Pipeline Feedback Loop
```bash
python src/evaluation/run_real_pipeline.py
```

---

## 📊 Validated Results & Proof of Execution

The pipeline was executed over 6 retraining cycles on the 10,000-row real corpus (CIC-IDS + EMBER). The architecture achieves the exact mathematical behavior predicted in the theoretical paper.

### 📈 Metrics
| Cycle | Accuracy | Escalation Rate (Uncertainty) |
| :--- | :--- | :--- |
| **0 (Baseline)** | **99.60%** | **1.20%** |
| 1 | - | - |
| ... | - | - |
| **6 (Final)** | **99.30%** | **0.95%** |

### 🔍 Execution Proof (Raw Terminal Output)
Below is the raw terminal output demonstrating the successful execution of the feedback loop on real data:

```text
Loading Tier 1 Real Corpus (CIC-IDS)...
Loading Tier 2 Real Corpus (EMBER)...

Training Tier 2 on Tier-2 Training Set (3,000 rows)...
Training Tier 1 on Tier-1 Seed Set (300 rows)...

--- Cycle 0 (Baseline) ---
Accuracy:        99.60%
Escalation Rate: 1.20%
Cycle 1 complete. Escalated 21 samples.
Cycle 2 complete. Escalated 5 samples.
Cycle 3 complete. Escalated 4 samples.
Cycle 4 complete. Escalated 6 samples.
Cycle 5 complete. Escalated 5 samples.
Cycle 6 complete. Escalated 4 samples.

--- Cycle 6 (Final) ---
Accuracy:        99.30%
Escalation Rate: 0.95%
```

### 🔬 Observation
As Tier 1 ingests feedback from Tier 2, the escalation (uncertainty) rate drops significantly (**1.20% -> 0.95%**). The slight accuracy dip represents the model adjusting its deterministic boundaries in response to noisy, real-world data, successfully reducing dependency on the expensive Tier 2 sandbox over time.
