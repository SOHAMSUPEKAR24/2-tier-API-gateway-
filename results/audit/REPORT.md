# Phase 1 Audit Report: Result Integrity & Data Leakage

**Date:** 2026-10-18  
**Experiment:** Repository Integrity Audit & Dst Port Leakage Verification  
**Config:** `configs/experiment_base.yaml`  
**Seeds Tested:** 42, 71, 137, 256, 512  

---

## 1. What Was Run
1. **Repository-Wide Metric Audit:**
   - Scanned all codebase files (Python scripts, TSX dashboard files, Next.js routes) for hardcoded metric constants, artificial latency delays, and simulated curves.
   - Categorized all metrics into `MEASURED`, `HARDCODED`, `SIMULATED`, or `CLEAN`.
   - Results documented in `docs/audit.md` and serialized to `results/audit/20261018_122239/result.json`.

2. **Feature Leakage Experiment (Dst Port):**
   - Evaluated Tier 1 Random Forest classifier (40 trees, depth 6) across 5 random seeds on `data/generated/tier1_real_corpus.csv` with and without the `Dst Port` identifier column.
   - Enforced deduplication to prevent cross-split leakage.

3. **Tier 2 Label Source Investigation:**
   - Audited `src/evaluation/run_unseen_multi_attack_loop.py` to trace how escalated payloads are verified.
   - Documented in `docs/tier2_label_source.md`.

---

## 2. Quantitative Results

### Feature Leakage Comparison (Dst Port)

| Condition | Test Accuracy (mean ± 95% CI) | Escalation Rate (mean ± 95% CI) |
| :--- | :--- | :--- |
| **WITH `Dst Port` (Leaky Identifier)** | **99.96% ± 0.08%** | **0.14% ± 0.18%** |
| **WITHOUT `Dst Port` (Feature Allowlist)** | **99.85% ± 0.07%** | **0.04% ± 0.08%** |
| **Delta (With - Without)** | **+0.11%** | **+0.10%** |

---

## 3. Key Findings & Surprises

1. **Severe Pre-Existing Leakage Through Row Duplication:**
   When testing raw splits of `tier1_real_corpus.csv`, the test set contained **143 duplicate rows** that also appeared in the training set. This artificially inflated test accuracy. Strict deduplication is mandatory across all benchmark splits.

2. **Dst Port Impact:**
   The accuracy difference between including and excluding `Dst Port` on this specific balanced corpus is modest (+0.11%), but excluding it provides guaranteed immunity against port-based over-fitting (e.g., classifying all port 21 traffic as FTP-BruteForce).

3. **Discovered Hardcoded Values in Dashboard & Scripts:**
   - `src/dashboard/simulator.py`: Hardcoded cycle convergence profiles (`52.4% -> 98.9%`).
   - `src/dashboard/app.py`: Hardcoded feature importance radar coordinates `[0.35, 0.25, 0.15, ...]`.
   - `src/evaluation/run_unseen_multi_attack_loop.py`: Artificial latency addition `+ 65.0 ms`.
   - `src/next-dashboard/src/app/page.tsx` & `Charts.tsx`: Hardcoded fallback values (`0.35ms`, `78.5ms`, `1.15ms`).
   These are marked for complete replacement with dynamic reads from `results/*.json`.

4. **Tier 2 Feedback Mechanism Discrepancy:**
   The prior unseen-attack script called `tier2_rf.predict()` on random EMBER features, but secretly injected dataset ground truth (`batch_y_t1`) into Tier 1 retraining pool. This has been fully documented in `docs/tier2_label_source.md`.
