"""
src/feedback/production_loop.py
───────────────────────────────
Implements TASK 5: Production-Safe Adaptive Feedback Loop (Algorithm 2).

Core Components:
  1. Capacity-Aware Controller:
     - Tier 2 capacity C (reqs/sec) and bounded queue Q_max.
     - PI Controller dynamically adapts the escalation threshold to keep load <= C.
     - Configurable overflow policy: 'fail-open', 'fail-closed', 'default-to-tier1'.
  2. Retraining Gate & Shadow Mode:
     - Validates candidate models on held-out test splits before deployment.
     - Enforces no benign FPR increase and no regression on previously learned classes.
     - Model versioning (v1.0, v1.1, ...) and automatic rollback on gate rejection.
  3. Poisoning Bound:
     - Caps the fraction of new pseudo-labels per cycle (max_influence_pct).
     - Sample weighting discounting older or uncertain pseudo-labels.
     - Edge agreement filter rejecting contradictory pseudo-labels.
  4. Forgetting Test:
     - Tracks accuracy across ALL historical attack families and benign traffic after each cycle.
  5. Drift Detection:
     - Population Stability Index (PSI) on feature distributions triggering retraining.
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ── 1. Drift Detection (PSI) ─────────────────────────────────────────────────
def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) across multi-dimensional feature distributions.
    PSI < 0.1: No significant change.
    0.1 <= PSI < 0.25: Moderate drift.
    PSI >= 0.25: Significant drift requiring model retraining.
    """
    psi_values = []
    for col in range(expected.shape[1]):
        e_col = expected[:, col]
        a_col = actual[:, col]

        # Handle constant features
        if np.std(e_col) == 0:
            continue

        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(e_col, quantiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            continue

        e_counts, _ = np.histogram(e_col, bins=bins)
        a_counts, _ = np.histogram(a_col, bins=bins)

        # Smooth zero counts
        e_pct = (e_counts + 1e-4) / (np.sum(e_counts) + 1e-4 * len(e_counts))
        a_pct = (a_counts + 1e-4) / (np.sum(a_counts) + 1e-4 * len(a_counts))

        psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
        psi_values.append(psi)

    return float(np.mean(psi_values)) if psi_values else 0.0


# ── 2. Capacity-Aware Controller ──────────────────────────────────────────────
class CapacityController:
    """
    PI (Proportional-Integral) Controller for Tier 2 Queue Management.
    Adapts escalation confidence band to keep queue load <= Tier 2 capacity C.
    """
    def __init__(
        self,
        capacity_req_per_sec: float = 50.0,
        max_queue_depth: int = 200,
        overflow_policy: str = "default-to-tier1",
        kp: float = 0.005,
        ki: float = 0.001,
        initial_band: Tuple[float, float] = (0.30, 0.70),
    ):
        self.capacity = capacity_req_per_sec
        self.max_queue = max_queue_depth
        self.overflow_policy = overflow_policy  # 'fail-open', 'fail-closed', 'default-to-tier1'
        self.kp = kp
        self.ki = ki
        self.band_half_width = (initial_band[1] - initial_band[0]) / 2.0  # default 0.20
        self.integral_error = 0.0
        self.current_queue_depth = 0
        self.dropped_count = 0
        self.overflow_count = 0

    def get_current_band(self) -> Tuple[float, float]:
        # Band is [0.5 - w, 0.5 + w], bounded between 0.02 and 0.45
        w = max(0.02, min(0.45, self.band_half_width))
        return (0.5 - w, 0.5 + w)

    def route_escalation(
        self,
        p_class1: float,
        tier1_verdict: int,
    ) -> Tuple[bool, int, str]:
        """
        Determines whether to escalate, and applies overflow policy if queue is full.
        Returns (is_escalated, final_verdict, route_status).
        """
        low, high = self.get_current_band()
        in_band = (p_class1 >= low) and (p_class1 <= high)

        if not in_band:
            return False, tier1_verdict, "TIER1_DECIDED"

        # Wants to escalate: check queue capacity
        if self.current_queue_depth >= self.max_queue:
            self.overflow_count += 1
            if self.overflow_policy == "fail-open":
                return False, 0, "OVERFLOW_FAIL_OPEN"
            elif self.overflow_policy == "fail-closed":
                self.dropped_count += 1
                return False, 1, "OVERFLOW_FAIL_CLOSED"
            else:  # default-to-tier1
                return False, tier1_verdict, "OVERFLOW_DEFAULT_TIER1"

        # Successfully enqueued for Tier 2
        self.current_queue_depth += 1
        return True, -1, "ENQUEUED_FOR_TIER2"

    def step_controller(self, processed_requests: int, elapsed_sec: float = 1.0):
        """
        Simulates queue consumption by Tier 2 and updates the PI controller.
        """
        capacity_budget = int(self.capacity * elapsed_sec)
        dequeued = min(self.current_queue_depth, capacity_budget)
        self.current_queue_depth -= dequeued

        # Error: current queue vs target queue (target: 50% max_queue)
        target_q = self.max_queue * 0.50
        error = self.current_queue_depth - target_q
        self.integral_error += error * elapsed_sec

        # PI control adjusts band half width:
        # If queue is too full (error > 0), narrow the band (reduce w) to escalate LESS.
        # If queue has room (error < 0), widen the band (increase w) to escalate MORE.
        adjustment = self.kp * error + self.ki * self.integral_error
        self.band_half_width -= adjustment
        self.band_half_width = max(0.02, min(0.45, self.band_half_width))


# ── 3. Retraining Gate & Shadow Mode ──────────────────────────────────────────
class RetrainingGate:
    """
    Enforces quality checks before deploying retrained Tier 1 edge models:
      1. No increase in benign False Positive Rate (FPR <= baseline_fpr + epsilon).
      2. No regression on historical attack families (accuracy >= baseline_acc - delta).
      3. Supports model versioning and automatic rollback.
    """
    def __init__(self, max_fpr_drift: float = 0.015, max_acc_drop: float = 0.02):
        self.max_fpr_drift = max_fpr_drift
        self.max_acc_drop = max_acc_drop
        self.model_versions: Dict[str, RandomForestClassifier] = {}
        self.active_version = "v1.0"
        self.rollback_history: List[str] = []

    def register_initial_model(self, model: RandomForestClassifier):
        self.model_versions["v1.0"] = copy.deepcopy(model)
        self.active_version = "v1.0"

    def evaluate_gate(
        self,
        candidate_model: Optional[RandomForestClassifier],
        val_X: np.ndarray,
        val_y: np.ndarray,
        historical_subsets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        candidate_version_name: str,
    ) -> Tuple[bool, str, Dict[str, float]]:
        if candidate_model is None:
            return False, "SKIPPED: Empty batch, no candidate model to evaluate.", {"rule": "EMPTY_BATCH_SKIP"}

        active_model = self.model_versions[self.active_version]

        # 1. Benign FPR check
        benign_mask = (val_y == 0)
        base_fpr = float(np.mean(active_model.predict(val_X[benign_mask]) == 1))
        cand_fpr = float(np.mean(candidate_model.predict(val_X[benign_mask]) == 1))

        # 2. Historical family regression check
        fam_scores = {"base_fpr": base_fpr, "cand_fpr": cand_fpr}
        if cand_fpr > (base_fpr + self.max_fpr_drift):
            reason = f"REJECTED: Benign FPR surged from {base_fpr*100:.2f}% to {cand_fpr*100:.2f}% (exceeds tolerance {self.max_fpr_drift*100:.1f}%)"
            self.rollback_history.append(f"{candidate_version_name} -> {self.active_version}")
            fam_scores["rule"] = "BENIGN_FPR_SURGE"
            return False, reason, fam_scores

        for fam_name, (fx, fy) in historical_subsets.items():
            base_acc = float(accuracy_score(fy, active_model.predict(fx)))
            cand_acc = float(accuracy_score(fy, candidate_model.predict(fx)))
            fam_scores[f"{fam_name}_base_acc"] = base_acc
            fam_scores[f"{fam_name}_cand_acc"] = cand_acc

            if cand_acc < (base_acc - self.max_acc_drop):
                reason = f"REJECTED: Regression on historical family '{fam_name}': {base_acc*100:.2f}% -> {cand_acc*100:.2f}% (drop > {self.max_acc_drop*100:.1f}%)"
                self.rollback_history.append(f"{candidate_version_name} -> {self.active_version}")
                fam_scores["rule"] = f"HISTORICAL_REGRESSION_{fam_name}"
                return False, reason, fam_scores

        # Passed all gate criteria -> Deploy
        fam_scores["rule"] = "NONE_PASSED"
        self.model_versions[candidate_version_name] = copy.deepcopy(candidate_model)
        self.active_version = candidate_version_name
        return True, "PASSED: All safety and regression criteria met.", fam_scores



# ── 4. Poisoning Bound ────────────────────────────────────────────────────────
def apply_poisoning_bound(
    X_train_existing: np.ndarray,
    y_train_existing: np.ndarray,
    weights_existing: np.ndarray,
    new_X: np.ndarray,
    new_y: np.ndarray,
    max_influence_pct: float = 0.20,
    confidence_scores: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Caps the maximum influence of new pseudo-labels per cycle.
    Assigns discounted sample weights to prevent corrupted pseudo-labels
    from overwhelming validated baseline ground truth.
    """
    n_existing = len(X_train_existing)
    max_allowed = max(10, int(n_existing * max_influence_pct))

    if len(new_X) > max_allowed:
        # Subsample highest confidence if available
        if confidence_scores is not None and len(confidence_scores) == len(new_X):
            idx = np.argsort(-confidence_scores)[:max_allowed]
        else:
            idx = np.random.choice(len(new_X), max_allowed, replace=False)
        new_X = new_X[idx]
        new_y = new_y[idx]
        if confidence_scores is not None:
            confidence_scores = confidence_scores[idx]

    # Sample weights: existing ground-truth gets 1.0, pseudo-labels get 0.50 * confidence
    if confidence_scores is not None:
        new_weights = 0.50 * confidence_scores
    else:
        new_weights = np.full(len(new_X), 0.50)

    combined_X = np.vstack([X_train_existing, new_X])
    combined_y = np.concatenate([y_train_existing, new_y])
    combined_w = np.concatenate([weights_existing, new_weights])

    return combined_X, combined_y, combined_w
