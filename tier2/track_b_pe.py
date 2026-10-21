"""
tier2/track_b_pe.py
───────────────────
Track B: File Upload Static PE Malware Verification.
Dataset: EMBER 2018 Static PE Feature Dataset.

STATIC ANALYSIS ONLY: Samples are NEVER dynamically executed.
Architecture:
  - Tier 1 Edge Classifier: 6 fast header/structure features
    (binary size, virtual size, has_signature, has_tls, num_sections, mean_section_entropy).
  - Tier 2 Deep Verification Model: Complete EMBER static vector
    (byte frequency histogram, 2D byte entropy histogram, strings metadata, section hashes).

Documentation Note:
  EMBER-2018 distributes pre-extracted feature dictionaries to protect researcher machines
  from hosting active malware binaries. In a production gateway pipeline, LIEF extracts
  these identical features statically from inbound uploads (system binaries vs. MalwareBazaar samples).
"""

import json
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def extract_track_b_tier1_features(record: dict) -> List[float]:
    """
    Tier 1 Fast PE Features (6 lightweight header / metadata dimensions).
    Extracted in under 1 millisecond without deep unpacking.
    """
    general = record.get("general", {})
    sections = record.get("section", {}).get("sections", [])

    size = float(general.get("size", 0))
    vsize = float(general.get("vsize", 0))
    has_sig = float(general.get("has_signature", 0))
    has_tls = float(general.get("has_tls", 0))
    num_sections = float(len(sections))

    entropies = [s.get("entropy", 0.0) for s in sections if isinstance(s, dict)]
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    return [size, vsize, has_sig, has_tls, num_sections, mean_entropy]


def extract_track_b_tier2_features(record: dict) -> np.ndarray:
    """
    Tier 2 Deep Verification Model Features:
    Complete EMBER static representation:
      - 6 Tier 1 header features
      - 256 Byte frequency histogram counts
      - 256 2D Byte entropy distribution bins
      - String statistics (numstrings, avglength, paths, urls, registry, MZ)
      - Section min/max entropy and virtual size ratio
    Total: ~525 dense static features.
    """
    t1_feats = extract_track_b_tier1_features(record)

    hist = record.get("histogram", [0.0] * 256)
    if len(hist) < 256:
        hist = hist + [0.0] * (256 - len(hist))
    else:
        hist = hist[:256]

    byteentropy = record.get("byteentropy", [0.0] * 256)
    if len(byteentropy) < 256:
        byteentropy = byteentropy + [0.0] * (256 - len(byteentropy))
    else:
        byteentropy = byteentropy[:256]

    strings = record.get("strings", {})
    str_feats = [
        float(strings.get("numstrings", 0)),
        float(strings.get("avglength", 0.0)),
        float(strings.get("printables", 0)),
        float(strings.get("entropy", 0.0)),
        float(strings.get("paths", 0)),
        float(strings.get("urls", 0)),
        float(strings.get("registry", 0)),
        float(strings.get("MZ", 0)),
    ]

    sections = record.get("section", {}).get("sections", [])
    entropies = [s.get("entropy", 0.0) for s in sections if isinstance(s, dict)]
    max_entropy = float(np.max(entropies)) if entropies else 0.0
    min_entropy = float(np.min(entropies)) if entropies else 0.0

    full_vec = t1_feats + hist + byteentropy + str_feats + [max_entropy, min_entropy]
    return np.array(full_vec, dtype=np.float32)


class TrackBPEVerifier:
    """
    Two-Tier Architecture for Static PE Binary Verification.
    """
    def __init__(
        self,
        t1_band: Tuple[float, float] = (0.30, 0.70),
        random_state: int = 42,
    ):
        self.t1_band = t1_band
        self.random_state = random_state
        self.tier1_model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=random_state)
        self.tier2_model = RandomForestClassifier(n_estimators=120, max_depth=16, random_state=random_state)

    def fit(self, records_train: List[dict]):
        X1 = np.array([extract_track_b_tier1_features(r) for r in records_train])
        X2 = np.array([extract_track_b_tier2_features(r) for r in records_train])
        y = np.array([int(r.get("label", 0)) for r in records_train])

        self.tier1_model.fit(X1, y)
        self.tier2_model.fit(X2, y)

    def evaluate(self, records_test: List[dict]) -> Dict[str, float]:
        X1 = np.array([extract_track_b_tier1_features(r) for r in records_test])
        X2 = np.array([extract_track_b_tier2_features(r) for r in records_test])
        y = np.array([int(r.get("label", 0)) for r in records_test])

        # Measure Tier 1 latency
        t0 = time.perf_counter()
        probs1 = self.tier1_model.predict_proba(X1)
        t1_lat_ms = (time.perf_counter() - t0) * 1000.0 / len(X1)

        p1_class1 = probs1[:, 1]
        preds1 = (p1_class1 >= 0.5).astype(int)

        # Escalation mask
        esc_mask = (p1_class1 >= self.t1_band[0]) & (p1_class1 <= self.t1_band[1])
        esc_rate = float(np.mean(esc_mask))

        # Measure Tier 2 latency on a sample
        sample_x2 = X2[:1]
        t0 = time.perf_counter()
        for _ in range(50):
            _ = self.tier2_model.predict(sample_x2)
        t2_lat_ms = ((time.perf_counter() - t0) * 1000.0) / 50.0

        # Two-tier effective decision
        final_preds = preds1.copy()
        if np.any(esc_mask):
            final_preds[esc_mask] = self.tier2_model.predict(X2[esc_mask])

        blended_lat_ms = (1.0 - esc_rate) * t1_lat_ms + esc_rate * (t1_lat_ms + t2_lat_ms)

        acc_t1_only = float(accuracy_score(y, preds1))
        acc_two_tier = float(accuracy_score(y, final_preds))
        f1_two_tier = float(f1_score(y, final_preds, zero_division=0))
        prec_two_tier = float(precision_score(y, final_preds, zero_division=0))
        rec_two_tier = float(recall_score(y, final_preds, zero_division=0))

        tier2_preds_all = self.tier2_model.predict(X2)
        acc_t2_only = float(accuracy_score(y, tier2_preds_all))

        return {
            "tier1_only_accuracy": acc_t1_only,
            "two_tier_accuracy": acc_two_tier,
            "tier2_only_accuracy": acc_t2_only,
            "precision": prec_two_tier,
            "recall": rec_two_tier,
            "f1": f1_two_tier,
            "escalation_rate": esc_rate,
            "tier1_latency_p50_ms": t1_lat_ms,
            "tier2_latency_p50_ms": t2_lat_ms,
            "blended_latency_p50_ms": blended_lat_ms,
        }

    def evaluate_pareto(self, records_test: List[dict], budgets: List[float] = None) -> Dict[float, Dict[str, float]]:
        """
        Sweeps target escalation budgets from 0% to 100% to trace the complete
        Pareto frontier: Accuracy and Silent-Failure Rate (SFR) vs Escalation Rate (%)
        and vs Mean / Blended Latency (ms).
        """
        if budgets is None:
            budgets = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

        X1 = np.array([extract_track_b_tier1_features(r) for r in records_test])
        X2 = np.array([extract_track_b_tier2_features(r) for r in records_test])
        y = np.array([int(r.get("label", 0)) for r in records_test])

        # Measure Tier 1 latency
        t0 = time.perf_counter()
        probs1 = self.tier1_model.predict_proba(X1)
        t1_lat_ms = (time.perf_counter() - t0) * 1000.0 / len(X1)

        p1_class1 = probs1[:, 1]
        preds1 = (p1_class1 >= 0.5).astype(int)
        uncertainty = 1.0 - 2.0 * np.abs(p1_class1 - 0.5)

        # Measure Tier 2 latency
        sample_x2 = X2[:1]
        t0 = time.perf_counter()
        for _ in range(50):
            _ = self.tier2_model.predict(sample_x2)
        t2_lat_ms = ((time.perf_counter() - t0) * 1000.0) / 50.0

        tier2_preds_all = self.tier2_model.predict(X2)
        acc_t2_only = float(accuracy_score(y, tier2_preds_all) * 100.0)
        bacc_t2_only = float(balanced_accuracy_score(y, tier2_preds_all) * 100.0)
        acc_t1_only = float(accuracy_score(y, preds1) * 100.0)
        bacc_t1_only = float(balanced_accuracy_score(y, preds1) * 100.0)

        n_attacks = np.sum(y == 1)
        pareto_results = {}

        for b in budgets:
            if b <= 0.0:
                esc_mask = np.zeros(len(y), dtype=bool)
            elif b >= 1.0:
                esc_mask = np.ones(len(y), dtype=bool)
            else:
                tau_b = float(np.quantile(uncertainty, 1.0 - b))
                esc_mask = uncertainty >= tau_b

            final_preds = preds1.copy()
            final_preds[esc_mask] = tier2_preds_all[esc_mask]

            er = float(np.mean(esc_mask) * 100.0)
            acc = float(accuracy_score(y, final_preds) * 100.0)
            bacc = float(balanced_accuracy_score(y, final_preds) * 100.0)

            # Silent-Failure Rate: attack flows classified as benign without escalation
            silent_failures = np.sum((y == 1) & (preds1 == 0) & (~esc_mask))
            sfr = float(silent_failures / n_attacks * 100.0) if n_attacks > 0 else 0.0

            blended_lat_ms = (1.0 - er / 100.0) * t1_lat_ms + (er / 100.0) * (t1_lat_ms + t2_lat_ms)
            fpr = float(np.mean(final_preds[y == 0] == 1) * 100.0) if np.sum(y == 0) > 0 else 0.0
            rec = float(recall_score(y, final_preds, zero_division=0) * 100.0)
            prec = float(precision_score(y, final_preds, zero_division=0) * 100.0)
            f1 = float(f1_score(y, final_preds, zero_division=0) * 100.0)

            pareto_results[b] = {
                "budget": b,
                "escalation_rate": er,
                "two_tier_accuracy": acc,
                "balanced_accuracy": bacc,
                "silent_failure_rate": sfr,
                "blended_latency_p50_ms": blended_lat_ms,
                "fpr": fpr,
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "tier1_only_accuracy": acc_t1_only,
                "tier1_only_balanced_accuracy": bacc_t1_only,
                "tier2_only_accuracy": acc_t2_only,
                "tier2_only_balanced_accuracy": bacc_t2_only,
                "tier1_latency_p50_ms": t1_lat_ms,
                "tier2_latency_p50_ms": t2_lat_ms,
            }

        return pareto_results
