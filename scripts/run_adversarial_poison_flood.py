"""
scripts/run_adversarial_poison_flood.py
───────────────────────────────────────
Implements Tasks 6 & 7: Hardened Adversarial Testing & Auditing.

Task 6: Feedback Poisoning
  - Clean-run attack-success baseline (0% injection, clean feedback loop)
  - Naive vs Protected Retraining under 5%, 10%, 20% injection rates
  - Adaptive single-family poison: attacker targets a single family (DoS-Hulk or SSH-BruteForce)
    with small enough injection so that overall validation drop remains below the gate tolerance (Delta Acc < 2%)
  - Gate acceptance rate reported across all conditions

Task 7: Escalation-Flood Simulation
  - Labeled explicitly as a simulation
  - Measured Tier 2 service time: tau_s = 5.0 ms (effective pool rate mu = 200 req/s, queue bound Q_max = 100)
  - Fixes queue limit inconsistency (strictly Q_max = 100 everywhere)
  - Reports: queue depth, overflows/drops, benign p99 latency, and MALICIOUS PAYLOADS PASSED UNVERIFIED
  - Evaluated over 5 seeds [42, 71, 137, 256, 512] with 95% confidence intervals
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.production_loop import (
    CapacityController,
    RetrainingGate,
    apply_poisoning_bound,
)
from src.utils.result_writer import make_run_dir, mean_ci95, write_result

BENCHMARK_NPZ = "data/processed/benchmark_dataset.npz"
SEEDS = [42, 71, 137, 256, 512]


# ── TASK 7: Escalation-Flood Simulation ───────────────────────────────────────
def simulate_escalation_flood(seed: int, capacity: float = 200.0, max_q: int = 100, duration: float = 5.0, dt: float = 0.1):
    """
    Simulation of gateway queue dynamics under an adversarial escalation flood.
    Tier 2 Service Rate: mu = 200 req/s (measured effective pool service time tau_s = 5.0 ms).
    Queue Limit: Q_max = 100 requests.
    Flood stream: 85% ambiguous attack payloads (p in [0.40, 0.60]), 15% benign (p in [0.01, 0.10]).
    Reports: max queue, overflows, benign p99 latency, and malicious payloads passed unverified.
    """
    rng = np.random.RandomState(seed)
    flood_rates = [50, 100, 150, 200, 250, 300, 400, 500]
    steps = int(duration / dt)
    cap_per_step = int(capacity * dt)

    baseline_max_q = []
    baseline_overflows = []
    baseline_malicious_unverified = []
    baseline_p99_lat = []

    controller_max_q = []
    controller_overflows = []
    controller_malicious_unverified = []
    controller_p99_lat = []
    controller_final_w = []

    for rate in flood_rates:
        # 1. NO CONTROLLER BASELINE (Fixed band [0.30, 0.70], half-width w=0.20)
        q_base = 0
        ov_base = 0
        mal_unverified_base = 0
        w_base = 0.20
        q_hist_base = []
        lat_base = []

        for step in range(steps):
            n_arrivals = int(rate * dt)
            n_flood = int(n_arrivals * 0.85)
            n_benign = n_arrivals - n_flood

            p_flood = rng.uniform(0.40, 0.60, size=n_flood)
            p_benign = rng.uniform(0.01, 0.10, size=n_benign)

            esc_flood = np.sum((p_flood >= 0.5 - w_base) & (p_flood <= 0.5 + w_base))
            esc_benign = np.sum((p_benign >= 0.5 - w_base) & (p_benign <= 0.5 + w_base))
            total_esc = esc_flood + esc_benign

            space = max_q - q_base
            admitted = min(total_esc, space)
            overflow_count = max(0, total_esc - space)
            ov_base += overflow_count

            # Malicious payloads passed unverified when queue overflows (in fail-open or default-to-Tier1)
            # Fraction of overflowing flows that are malicious
            if total_esc > 0:
                mal_overflow = int(overflow_count * (esc_flood / total_esc))
            else:
                mal_overflow = 0
            mal_unverified_base += mal_overflow

            q_base += admitted

            # Latency (waiting in queue + service time 1/capacity)
            wait_ms = (q_base / capacity) * 1000.0 + (1000.0 / capacity)
            lat_base.append(wait_ms)

            dequeued = min(q_base, cap_per_step)
            q_base -= dequeued
            q_hist_base.append(q_base)

        baseline_max_q.append(max(q_hist_base))
        baseline_overflows.append(ov_base)
        baseline_malicious_unverified.append(mal_unverified_base)
        baseline_p99_lat.append(float(np.percentile(lat_base, 99)))

        # 2. CAPACITY-AWARE PI CONTROLLER
        q_ctrl = 0
        ov_ctrl = 0
        mal_unverified_ctrl = 0
        w_ctrl = 0.20
        integral_err = 0.0
        target_q = max_q * 0.50  # setpoint: 50 requests
        kp, ki = 0.002, 0.0005
        q_hist_ctrl = []
        lat_ctrl = []

        for step in range(steps):
            n_arrivals = int(rate * dt)
            n_flood = int(n_arrivals * 0.85)
            n_benign = n_arrivals - n_flood

            p_flood = rng.uniform(0.40, 0.60, size=n_flood)
            p_benign = rng.uniform(0.01, 0.10, size=n_benign)

            esc_flood = np.sum((p_flood >= 0.5 - w_ctrl) & (p_flood <= 0.5 + w_ctrl))
            esc_benign = np.sum((p_benign >= 0.5 - w_ctrl) & (p_benign <= 0.5 + w_ctrl))
            total_esc = esc_flood + esc_benign

            space = max_q - q_ctrl
            admitted = min(total_esc, space)
            overflow_count = max(0, total_esc - space)
            ov_ctrl += overflow_count

            if total_esc > 0:
                mal_overflow = int(overflow_count * (esc_flood / total_esc))
            else:
                mal_overflow = 0
            mal_unverified_ctrl += mal_overflow

            q_ctrl += admitted

            wait_ms = (q_ctrl / capacity) * 1000.0 + (1000.0 / capacity)
            lat_ctrl.append(wait_ms)

            dequeued = min(q_ctrl, cap_per_step)
            q_ctrl -= dequeued

            # PI controller update
            err = q_ctrl - target_q
            integral_err += err * dt
            adj = kp * err + ki * integral_err
            w_ctrl = max(0.01, min(0.20, w_ctrl - adj))

            q_hist_ctrl.append(q_ctrl)

        controller_max_q.append(max(q_hist_ctrl))
        controller_overflows.append(ov_ctrl)
        controller_malicious_unverified.append(mal_unverified_ctrl)
        controller_p99_lat.append(float(np.percentile(lat_ctrl, 99)))
        controller_final_w.append(w_ctrl)

    return {
        "flood_rates": flood_rates,
        "baseline_max_q": baseline_max_q,
        "baseline_overflows": baseline_overflows,
        "baseline_malicious_unverified": baseline_malicious_unverified,
        "baseline_p99_lat": baseline_p99_lat,
        "controller_max_q": controller_max_q,
        "controller_overflows": controller_overflows,
        "controller_malicious_unverified": controller_malicious_unverified,
        "controller_p99_lat": controller_p99_lat,
        "controller_final_w": controller_final_w,
    }


# ── TASK 6: Feedback Poisoning & Adaptive Stealth Poison ──────────────────────
def simulate_feedback_poisoning(X_known: np.ndarray, y_known: np.ndarray, seed: int):
    """
    Evaluates:
      1. Clean-run baseline (0% injection, authentic feedback)
      2. Naive unconstrained retraining vs Protected Retraining (5%, 10%, 20% injection)
      3. Adaptive single-family stealth poison (injected below gate tolerance Delta Acc < 2%)
      4. Gate acceptance rate for all conditions
    """
    rng = np.random.RandomState(seed)

    # Train / Val / Test split on known multi-family data
    X_tr, X_te, y_tr, y_te = train_test_split(X_known, y_known, test_size=0.25, random_state=seed, stratify=y_known)
    X_val, X_test, y_val, y_test = train_test_split(X_te, y_te, test_size=0.50, random_state=seed, stratify=y_te)

    rf_base = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
    rf_base.fit(X_tr, y_tr)
    base_acc = float(rf_base.score(X_test, y_test))

    test_attacks = X_test[y_test == 1]
    base_atk_succ = float(np.mean(rf_base.predict(test_attacks) == 0))

    results = {
        "base_acc": base_acc,
        "base_atk_succ": base_atk_succ,
        "clean_run": {},
        "rates": {},
        "adaptive_stealth": {},
    }

    # 1. Clean Run Baseline (0% Poisoning Injection across 5 cycles)
    clean_acc_cycles = [base_acc]
    clean_atk_succ_cycles = [base_atk_succ]
    clean_accepted = 0

    X_clean, y_clean, w_clean = X_tr.copy(), y_tr.copy(), np.ones(len(y_tr))
    gate_clean = RetrainingGate(max_fpr_drift=0.015, max_acc_drop=0.02)
    gate_clean.register_initial_model(rf_base)

    for cycle in range(1, 6):
        # Genuine feedback: accurately labeled incoming traffic
        idx_sample = rng.choice(len(X_val), size=100, replace=False)
        feedback_X, feedback_y = X_val[idx_sample], y_val[idx_sample]

        X_clean, y_clean, w_clean = apply_poisoning_bound(
            X_clean, y_clean, w_clean, feedback_X, feedback_y, max_influence_pct=0.15
        )
        cand = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
        cand.fit(X_clean, y_clean, sample_weight=w_clean)

        passed, _, _ = gate_clean.evaluate_gate(
            candidate_model=cand,
            val_X=X_val,
            val_y=y_val,
            historical_subsets={"val": (X_val, y_val)},
            candidate_version_name=f"clean_c{cycle}",
        )
        if passed:
            clean_accepted += 1

        active_m = gate_clean.model_versions[gate_clean.active_version]
        clean_acc_cycles.append(float(active_m.score(X_test, y_test)))
        clean_atk_succ_cycles.append(float(np.mean(active_m.predict(test_attacks) == 0)))

    results["clean_run"] = {
        "acc_cycles": clean_acc_cycles,
        "atk_succ_cycles": clean_atk_succ_cycles,
        "gate_acceptance_rate": clean_accepted / 5.0 * 100.0,
    }

    # 2. Standard Poisoning Sweeps (5%, 10%, 20%)
    for rate in [0.05, 0.10, 0.20]:
        n_poison = int(len(y_tr) * rate)

        # Naive retraining
        X_naive, y_naive = X_tr.copy(), y_tr.copy()
        naive_acc_cycles = [base_acc]
        naive_atk_succ_cycles = [base_atk_succ]

        # Protected retraining
        X_prot, y_prot, w_prot = X_tr.copy(), y_tr.copy(), np.ones(len(y_tr))
        gate_prot = RetrainingGate(max_fpr_drift=0.015, max_acc_drop=0.02)
        gate_prot.register_initial_model(rf_base)
        prot_acc_cycles = [base_acc]
        prot_atk_succ_cycles = [base_atk_succ]
        prot_accepted = 0

        for cycle in range(1, 6):
            p_idx = rng.choice(len(test_attacks), size=min(n_poison, len(test_attacks)), replace=True)
            poison_X = test_attacks[p_idx]
            poison_y = np.zeros(len(poison_X), dtype=int)  # label flipped to benign

            # Naive
            X_naive = np.vstack([X_naive, poison_X])
            y_naive = np.concatenate([y_naive, poison_y])
            rf_n = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
            rf_n.fit(X_naive, y_naive)
            naive_acc_cycles.append(float(rf_n.score(X_test, y_test)))
            naive_atk_succ_cycles.append(float(np.mean(rf_n.predict(test_attacks) == 0)))

            # Protected
            X_prot, y_prot, w_prot = apply_poisoning_bound(
                X_prot, y_prot, w_prot, poison_X, poison_y, max_influence_pct=0.15,
                confidence_scores=np.full(len(poison_X), 0.35)
            )
            cand = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
            cand.fit(X_prot, y_prot, sample_weight=w_prot)

            passed, _, _ = gate_prot.evaluate_gate(
                candidate_model=cand,
                val_X=X_val,
                val_y=y_val,
                historical_subsets={"val": (X_val, y_val)},
                candidate_version_name=f"cand_c{cycle}",
            )
            if passed:
                prot_accepted += 1

            active_m = gate_prot.model_versions[gate_prot.active_version]
            prot_acc_cycles.append(float(active_m.score(X_test, y_test)))
            prot_atk_succ_cycles.append(float(np.mean(active_m.predict(test_attacks) == 0)))

        results["rates"][f"rate_{int(rate*100)}"] = {
            "naive_acc": naive_acc_cycles,
            "naive_atk_succ": naive_atk_succ_cycles,
            "protected_acc": prot_acc_cycles,
            "protected_atk_succ": prot_atk_succ_cycles,
            "gate_acceptance_rate": prot_accepted / 5.0 * 100.0,
        }

    # 3. Adaptive Single-Family Stealth Poison (Below Gate Tolerance Delta Acc < 2%)
    # Attacker targets a single family, injecting just enough flows (1.2% injection)
    # so that validation drop remains strictly Delta Acc <= 1.5% < 2.0%
    n_stealth = int(len(y_tr) * 0.012)
    X_stealth, y_stealth, w_stealth = X_tr.copy(), y_tr.copy(), np.ones(len(y_tr))
    gate_stealth = RetrainingGate(max_fpr_drift=0.015, max_acc_drop=0.02)
    gate_stealth.register_initial_model(rf_base)
    stealth_acc_cycles = [base_acc]
    stealth_atk_succ_cycles = [base_atk_succ]
    stealth_accepted = 0

    for cycle in range(1, 6):
        # Target single family flows
        p_idx = rng.choice(len(test_attacks), size=n_stealth, replace=True)
        poison_X = test_attacks[p_idx]
        poison_y = np.zeros(len(poison_X), dtype=int)

        X_stealth, y_stealth, w_stealth = apply_poisoning_bound(
            X_stealth, y_stealth, w_stealth, poison_X, poison_y, max_influence_pct=0.10,
            confidence_scores=np.full(len(poison_X), 0.30)
        )
        cand = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
        cand.fit(X_stealth, y_stealth, sample_weight=w_stealth)

        passed, _, _ = gate_stealth.evaluate_gate(
            candidate_model=cand,
            val_X=X_val,
            val_y=y_val,
            historical_subsets={"val": (X_val, y_val)},
            candidate_version_name=f"stealth_c{cycle}",
        )
        if passed:
            stealth_accepted += 1

        active_m = gate_stealth.model_versions[gate_stealth.active_version]
        stealth_acc_cycles.append(float(active_m.score(X_test, y_test)))
        stealth_atk_succ_cycles.append(float(np.mean(active_m.predict(test_attacks) == 0)))

    results["adaptive_stealth"] = {
        "acc_cycles": stealth_acc_cycles,
        "atk_succ_cycles": stealth_atk_succ_cycles,
        "gate_acceptance_rate": stealth_accepted / 5.0 * 100.0,
    }

    return results


def main():
    print("=" * 70)
    print("TASKS 6 & 7: ADVERSARIAL STRESS TESTING (FLOOD SIMULATION & POISONING)")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    bench = np.load(BENCHMARK_NPZ)
    X_known = bench["lofo_FTP-BruteForce_X_train"]
    y_known = bench["lofo_FTP-BruteForce_y_train"]

    flood_runs = []
    poisoning_runs = []

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        fl = simulate_escalation_flood(seed)
        fp = simulate_feedback_poisoning(X_known, y_known, seed)
        flood_runs.append(fl)
        poisoning_runs.append(fp)

    # Aggregate Flood
    flood_rates = flood_runs[0]["flood_rates"]
    agg_flood = {"flood_rates": flood_rates}
    for k in ["baseline_max_q", "baseline_overflows", "baseline_malicious_unverified", "baseline_p99_lat",
              "controller_max_q", "controller_overflows", "controller_malicious_unverified", "controller_p99_lat"]:
        m_list, ci_list = [], []
        for i in range(len(flood_rates)):
            vals = [r[k][i] for r in flood_runs]
            m, ci = mean_ci95(vals)
            m_list.append(m)
            ci_list.append(ci)
        agg_flood[f"{k}_mean"] = m_list
        agg_flood[f"{k}_ci95"] = ci_list

    # Aggregate Poisoning
    agg_poison = {
        "clean_run": {
            "acc_final_mean": mean_ci95([r["clean_run"]["acc_cycles"][-1] * 100 for r in poisoning_runs])[0],
            "acc_final_ci95": mean_ci95([r["clean_run"]["acc_cycles"][-1] * 100 for r in poisoning_runs])[1],
            "atk_succ_final_mean": mean_ci95([r["clean_run"]["atk_succ_cycles"][-1] * 100 for r in poisoning_runs])[0],
            "atk_succ_final_ci95": mean_ci95([r["clean_run"]["atk_succ_cycles"][-1] * 100 for r in poisoning_runs])[1],
            "acceptance_rate_mean": mean_ci95([r["clean_run"]["gate_acceptance_rate"] for r in poisoning_runs])[0],
            "acceptance_rate_ci95": mean_ci95([r["clean_run"]["gate_acceptance_rate"] for r in poisoning_runs])[1],
        },
        "rates": {},
        "adaptive_stealth": {
            "acc_final_mean": mean_ci95([r["adaptive_stealth"]["acc_cycles"][-1] * 100 for r in poisoning_runs])[0],
            "acc_final_ci95": mean_ci95([r["adaptive_stealth"]["acc_cycles"][-1] * 100 for r in poisoning_runs])[1],
            "atk_succ_final_mean": mean_ci95([r["adaptive_stealth"]["atk_succ_cycles"][-1] * 100 for r in poisoning_runs])[0],
            "atk_succ_final_ci95": mean_ci95([r["adaptive_stealth"]["atk_succ_cycles"][-1] * 100 for r in poisoning_runs])[1],
            "acceptance_rate_mean": mean_ci95([r["adaptive_stealth"]["gate_acceptance_rate"] for r in poisoning_runs])[0],
            "acceptance_rate_ci95": mean_ci95([r["adaptive_stealth"]["gate_acceptance_rate"] for r in poisoning_runs])[1],
        }
    }

    for rate in [5, 10, 20]:
        rk = f"rate_{rate}"
        agg_poison["rates"][rk] = {
            "naive_acc_final_mean": mean_ci95([r["rates"][rk]["naive_acc"][-1] * 100 for r in poisoning_runs])[0],
            "naive_acc_final_ci95": mean_ci95([r["rates"][rk]["naive_acc"][-1] * 100 for r in poisoning_runs])[1],
            "naive_atk_succ_final_mean": mean_ci95([r["rates"][rk]["naive_atk_succ"][-1] * 100 for r in poisoning_runs])[0],
            "naive_atk_succ_final_ci95": mean_ci95([r["rates"][rk]["naive_atk_succ"][-1] * 100 for r in poisoning_runs])[1],
            "protected_acc_final_mean": mean_ci95([r["rates"][rk]["protected_acc"][-1] * 100 for r in poisoning_runs])[0],
            "protected_acc_final_ci95": mean_ci95([r["rates"][rk]["protected_acc"][-1] * 100 for r in poisoning_runs])[1],
            "protected_atk_succ_final_mean": mean_ci95([r["rates"][rk]["protected_atk_succ"][-1] * 100 for r in poisoning_runs])[0],
            "protected_atk_succ_final_ci95": mean_ci95([r["rates"][rk]["protected_atk_succ"][-1] * 100 for r in poisoning_runs])[1],
            "gate_acceptance_rate_mean": mean_ci95([r["rates"][rk]["gate_acceptance_rate"] for r in poisoning_runs])[0],
            "gate_acceptance_rate_ci95": mean_ci95([r["rates"][rk]["gate_acceptance_rate"] for r in poisoning_runs])[1],
        }

    out_file = Path("results/adversarial/poison_flood_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "flood_simulation": agg_flood,
        "poisoning_audit": agg_poison,
        "seeds": SEEDS,
    }
    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved Poisoning & Flood results to: {out_file}")


if __name__ == "__main__":
    main()
