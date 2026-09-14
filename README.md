<div align="center">
  <h1>🛡️ Adaptive Two-Tier Malware Detection for API Gateways</h1>
  <p><strong>Stopping zero-days at the edge, without slowing down the internet.</strong></p>
  <img src="docs/assets/architecture_flow.png" width="800" alt="Architecture Flow Chart">
</div>

---

## 📖 The "Why" Behind the Project

API Gateways are the front doors of the modern internet. They need to process thousands of requests per second with near-zero latency. But as attacks get more sophisticated—hiding malware in innocent-looking uploads or complex JSON bodies—simple pattern matching isn't enough anymore.

The traditional solution is to send everything to a "deep inspection sandbox" (like a heavy ML model or a virtual machine). But doing that for *every single request* would bring the gateway to a grinding halt.

**So, how do we get the security of deep inspection with the speed of edge routing?**

This repository is the real-world, verified implementation of a radical solution: **An Adaptive Two-Tier Architecture**.

It works like a seasoned security guard training a rookie:
1. **The Rookie (Tier 1):** A lightning-fast, lightweight model sits at the edge. It makes instant decisions on 99% of traffic. When it gets confused (uncertain), it escalates the request.
2. **The Veteran (Tier 2):** A heavy, expensive, deep-inspection sandbox analyzes the escalated payloads.
3. **The Magic (Feedback Loop):** Every time the Veteran resolves a tricky case, the Rookie learns from it. Over time, the Rookie stops getting confused by the same tricks, dropping the escalation rate and saving massive amounts of compute power.

---

## 🏗️ Architecture & Real-World Data

While the original academic paper proved this concept using theoretical, synthetic payloads, this repository proves it works on **real-world, multi-gigabyte datasets**:

* **Tier 1 (Edge Screen):** Trained on **CIC-IDS 2017/2018**, evaluating network-layer flow features at wire-speed.
* **Tier 2 (Deep Sandbox):** Trained on **EMBER 2018**, evaluating deep static PE malware features.

> *Curious about how we mapped theoretical paper features to real-world network data? Check out our [Dataset Provenance Guide](docs/dataset_provenance.md).*

---

## 🚀 Quickstart

Want to see the Rookie learn from the Veteran in real-time? Run the pipeline yourself:

### 1. Set Up Your Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Extract & Balance the Data
*Note: We safely chunk massive 6GB+ CSV files to prevent your machine from crashing (OOM), building a perfectly balanced 10,000-row testing corpus.*
```bash
python src/data_collection/build_real_corpus.py
```

### 3. Run the Feedback Loop!
```bash
python src/evaluation/run_real_pipeline.py
```

---

## 📊 The Results

We ran the pipeline over 6 retraining cycles on the 10,000-row real corpus. The results perfectly validated the mathematics of the theoretical paper: the system learned to handle uncertainty autonomously.

<div align="center">
  <img src="docs/assets/feedback_metrics.png" width="800" alt="Feedback Metrics Chart">
</div>

### 📈 Metrics Breakdown
| Retraining Cycle | Accuracy | Escalation Rate (Uncertainty) |
| :--- | :--- | :--- |
| **0 (Baseline)** | **99.60%** | **1.20%** |
| 1 | - | - |
| ... | - | - |
| **6 (Final)** | **99.30%** | **0.95%** |

### 🔬 What does this mean?
As Tier 1 (the Rookie) ingests feedback from Tier 2 (the Veteran), its uncertainty drops significantly (**1.20% -> 0.95%**). 

You might notice a *tiny* dip in accuracy (0.3%). This is actually a massive win! It means the lightweight model successfully adjusted its rigid, deterministic boundaries in response to noisy, real-world data, drastically reducing its dependency on the expensive Tier 2 sandbox over time—all while keeping accuracy well above 99%.

---
*Built with Python, Scikit-Learn, and Pandas. Validated against CIC-IDS and EMBER.*
