# Model and Data Card

## Model Overview
- **Model Name:** Adaptive Two-Tier Malware Detection API Gateway (Edge Screen + Deep Verifier)
- **Architecture:** Two-Tier Hierarchical Inference
  - **Tier 1 (Edge Screen):** Random Forest (40 trees, max_depth=6) / Evidential Deep Neural Network evaluating 9 allowlisted flow/payload features. Operating latency: $\le 0.05\text{ ms}$.
  - **Tier 2 (Deep Verification Model):** TF-IDF Character 3-5 gram Vectorizer (1,500 features) + Random Forest (120 trees) or dense static PE representation (525 features). Operating latency: $1.5 - 20\text{ ms}$.
- **Escalation Triggers:**
  - Probability Band $[0.30, 0.70]$
  - Split-Conformal Prediction ($\alpha = 0.10$)
  - Isolation Forest Novelty Detection
  - Bagged Tree Ensemble Variance
  - Evidential Deep Learning (EDL) with Warm-Start Dirichlet Prior

## Training Data & Provenance
- **Network Flows:** CSE-CIC-IDS2018 / CIC-IDS2017 (Canadian Institute for Cybersecurity). Strictly filtered through canonical feature allowlist; all identifier columns (`Dst Port`, `Protocol`, `Timestamp`, `Flow ID`, `Src/Dst IP`) stripped.
- **HTTP Payloads (Track A):** CSIC 2010 HTTP Dataset (Spanish Research Council). 61,065 genuine HTTP requests.
- **Static PE Uploads (Track B):** EMBER 2018 Static PE Dataset. Static analysis only; raw binaries never executed.

## Evaluation Protocols
- **Protocol 1 (Novel Attack Stream):** Trained strictly on Benign + FTP-BruteForce; tested against held-out Benign and 4 novel attack families (`SSH-BruteForce`, `DoS-GoldenEye`, `DoS-Hulk`, `DoS-Slowloris`).
- **Protocol 2 (Leave-One-Family-Out / LOFO):** Held-out family evaluation across all 5 classes.
- **Protocol 3 (Prevalence Ratios):** Evaluation under realistic edge base rates (1:100 and 1:1000 attack-to-benign prevalence).

## Intended Use & Safety Controls
- **Intended Deployment:** Reverse proxy API gateway edge (FastAPI / Envoy / NGINX / Kong).
- **Safety Mechanisms:**
  - Capacity-Aware PI Controller keeping Tier 2 load bounded.
  - Retraining Gate enforcing benign FPR $\le \text{baseline} + 1.5\%$ and historical class accuracy retention.
  - Poisoning Bound capping per-cycle influence to $\le 15-20\%$.
  - Continuous drift detection via Population Stability Index (PSI).
