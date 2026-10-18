# Retraining Gate Audit & Decision Log

## Overview
This document audits the behavior, trigger criteria, and decision mechanics of the **Retraining Gate (Algorithm 2, Step 5)** in the adaptive feedback loop.

---

## 1. Root Cause Analysis: Legacy Cycle 1 Rejection Artifact

### The Observed Anomaly
In earlier experimental iterations, Cycle 1 was logged as `REJECTED_ROLLBACK` across all 5 random seeds (`42, 71, 137, 256, 512`), while Cycles 2 through 5 passed validation.

### Root Cause Investigation
1. **Array Ordering in Benchmark Dataset:**
   In `data/processed/benchmark_dataset.npz`, the `ftp_only_X_test` matrix was constructed by appending the benign test flows first, followed by the novel attack family flows.
2. **Sequential Batching Without Shuffling:**
   In the legacy adaptation script, `X_stream` was sliced sequentially into 5 chronological chunks (`cycle_size = len(X_stream) // 5` $\approx 1,971$ samples).
   - Slice 1 (Cycle 1) consisted entirely of the first 1,971 flows, which were **pure in-distribution benign traffic**.
   - Because the initial baseline model `v1.0` was already trained on benign traffic, its predicted attack probabilities for every flow in Slice 1 were confident benign ($p < 0.10$).
   - As a result, **zero samples** fell inside the uncertainty escalation band $[0.30, 0.70]$:
     $$\text{len}(\text{escalated\_indices}) = 0$$
3. **The Reporting Bug:**
   The legacy code contained the following logic:
   ```python
   if len(escalated_indices) > 0:
       passed, gate_reason, metrics = gate.evaluate_gate(...)
   else:
       passed = False
       gate_reason = "SKIPPED: Zero escalations in batch"
   
   gate_status = "PASSED" if passed else "REJECTED_ROLLBACK"
   ```
   When `escalated_indices` was empty, `passed` was assigned `False`, causing the status to be reported as `REJECTED_ROLLBACK` even though **no candidate model was ever trained or rejected**.

### Resolution Implemented
1. **Status Disambiguation:**
   Cycles where zero escalations occur are now properly recorded as `SKIPPED_NO_DATA` with reason `"SKIPPED: Zero escalations in batch (familiar traffic)"`.
2. **Realistic Stream Shuffling:**
   `X_stream` is now randomly permuted using the experiment seed prior to slicing, simulating realistic production network traffic containing an authentic mixture of background benign requests and emerging novel attack flows across all cycles.

---

## 2. Retraining Gate Safety Criteria

The Retraining Gate enforces two non-negotiable safety invariant checks before allowing any retrained Tier 1 candidate model $\mathcal{M}_{\text{cand}}$ to replace the currently deployed active model $\mathcal{M}_{\text{active}}$:

### Invariant 1: Benign False Positive Rate (FPR) Bounding
$$\text{FPR}_{\text{cand}} \le \text{FPR}_{\text{active}} + \epsilon_{\text{FPR}} \quad (\epsilon_{\text{FPR}} = 1.5\%)$$
- Evaluated on a strictly held-out validation set of benign traffic $\mathcal{D}_{\text{val}}^{\text{benign}}$.
- Prevents retraining from corrupting edge classification of legitimate user and client requests.

### Invariant 2: Anti-Regression on Historical Attack Classes
$$\text{Acc}_{\text{cand}}(f) \ge \text{Acc}_{\text{active}}(f) - \delta_{\text{acc}} \quad (\delta_{\text{acc}} = 3.0\%) \quad \forall f \in \mathcal{F}_{\text{history}}$$
- Evaluated on dedicated held-out test splits for all previously encountered attack families (`FTP-BruteForce`, `SSH-BruteForce`, `DoS-GoldenEye`, `DoS-Hulk`, `DoS-Slowloris`).
- Prevents catastrophic forgetting: an edge model retrained on a new attack family must not forget how to detect earlier families.

---

## 3. Deliberate Adversarial Poisoning Test (Cycle 3 Audit)

To empirically validate that the Retraining Gate actively rejects corrupted candidate models under live attack:
1. **Attack Injection:** In Cycle 3, an adversary injects 180 attack flows mislabeled with inverted labels ($y=0$, claiming benign status) to execute an adversarial forgetting exploit targeting `FTP-BruteForce`.
2. **Candidate Training:** Candidate model `cand_v1.3` is fitted on the poisoned training pool.
3. **Gate Evaluation:**
   - Candidate benign FPR is checked: $0.00\% \to 0.00\%$ (passes Invariant 1).
   - Historical family regression is checked:
     - `FTP-BruteForce`: Baseline accuracy $= 99.81\%$, Candidate accuracy $= 85.55\%$ ($\Delta = -14.26\%$, exceeding $\delta_{\text{acc}} = 3.0\%$).
4. **Gate Action:**
   - **Decision:** `REJECTED_ROLLBACK`
   - **Triggering Rule:** `HISTORICAL_REGRESSION_FTP-BruteForce`
   - **Action Taken:** Candidate `cand_v1.3` is discarded. Active deployment remains pinned to verified model `v1.2`.
5. **Subsequent Recovery:** In Cycles 4 and 5, clean streams arrive. Candidate models satisfy all safety invariants and successfully deploy as `v1.4` and `v1.5`.

---

## 4. Cycle Decision Summary

| Cycle | Input Stream | Escalated Count | Gate Decision | Triggering Rule | Active Version |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Initial Training Set | N/A (Baseline) | **DEPLOYED** | `Baseline Registration` | `v1.0` |
| **1** | Novel Stream Chunk 1 | >0 | **PASSED** | `NONE_PASSED` | `v1.1` |
| **2** | Novel Stream Chunk 2 | >0 | **PASSED** | `NONE_PASSED` | `v1.2` |
| **3** | Deliberate Poisoned Batch | >0 (Poisoned) | **REJECTED_ROLLBACK** | `HISTORICAL_REGRESSION_FTP-BruteForce` | `v1.2` (Rolled Back) |
| **4** | Novel Stream Chunk 4 | >0 | **PASSED** | `NONE_PASSED` | `v1.4` |
| **5** | Novel Stream Chunk 5 | >0 | **PASSED** | `NONE_PASSED` | `v1.5` |
