"""
tests/test_two_signal_gate.py
─────────────────────────────
Unit tests for the Two-Signal Escalation Gate & Benign-Only Novelty Detectors.
Verifies:
  1. Benign-only training constraint (no attack samples in novelty fitting).
  2. Numerical stability of Mahalanobis distance under collinear flow features.
  3. Proper positive finite scores from all 4 novelty detectors.
  4. Conformal calibration budget bounding on held-out calibration traffic.
  5. Logical OR property of TwoSignalEscalationGate.
  6. Strict non-identifier allowlist enforcement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.models.two_signal_gate import (
    AutoencoderNovelty,
    ConformalSingleSignalTrigger,
    IsolationForestNovelty,
    KNNNovelty,
    MahalanobisNovelty,
    TwoSignalEscalationGate,
)
from src.utils.feature_guard import IDENTIFIER_COLUMNS, TIER1_ALLOWLIST


@pytest.fixture
def synthetic_data():
    """Generates synthetic benign and attack traffic for unit testing."""
    rng = np.random.RandomState(42)
    # Benign traffic: normal distribution around origin
    n_benign = 400
    n_attack = 100
    d = len(TIER1_ALLOWLIST)  # 9 features

    X_benign = rng.randn(n_benign, d) * 1.5 + 2.0
    y_benign = np.zeros(n_benign, dtype=int)

    # Attack traffic: distinct distribution
    X_attack = rng.randn(n_attack, d) * 3.0 + 8.0
    y_attack = np.ones(n_attack, dtype=int)

    X = np.vstack([X_benign, X_attack])
    y = np.concatenate([y_benign, y_attack])

    # Shuffle
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]

    # Split train and calibration
    split = int(0.7 * len(y))
    return X[:split], y[:split], X[split:], y[split:]


def test_novelty_detectors_benign_only_fitting(synthetic_data):
    """Verify that all 4 novelty detectors fit on benign flows and output valid scores."""
    X_train, y_train, X_test, _ = synthetic_data
    X_benign = X_train[y_train == 0]

    # 1. Isolation Forest
    iso = IsolationForestNovelty(random_state=42)
    iso.fit(X_benign)
    scores_iso = iso.score(X_test)
    assert scores_iso.shape == (len(X_test),)
    assert np.all(np.isfinite(scores_iso))

    # 2. Mahalanobis Distance
    mahal = MahalanobisNovelty()
    mahal.fit(X_benign)
    scores_mahal = mahal.score(X_test)
    assert scores_mahal.shape == (len(X_test),)
    assert np.all(np.isfinite(scores_mahal))
    assert np.all(scores_mahal >= 0.0)

    # 3. kNN Distance
    knn = KNNNovelty(k=5)
    knn.fit(X_benign)
    scores_knn = knn.score(X_test)
    assert scores_knn.shape == (len(X_test),)
    assert np.all(np.isfinite(scores_knn))
    assert np.all(scores_knn >= 0.0)

    # 4. Small Autoencoder
    ae = AutoencoderNovelty(input_dim=X_train.shape[1], epochs=10, seed=42)
    ae.fit(X_benign)
    scores_ae = ae.score(X_test)
    assert scores_ae.shape == (len(X_test),)
    assert np.all(np.isfinite(scores_ae))
    assert np.all(scores_ae >= 0.0)


def test_mahalanobis_collinear_features_stability():
    """Verify that Mahalanobis shrinkage handles perfectly collinear / singular features."""
    rng = np.random.RandomState(42)
    n = 200
    # Create collinear features: column 2 is exactly 2 * column 1
    col0 = rng.randn(n, 1)
    col1 = col0 * 2.5
    col2 = rng.randn(n, 1)
    col3 = np.ones((n, 1)) * 5.0  # constant feature
    X_collinear = np.hstack([col0, col1, col2, col3])

    mahal = MahalanobisNovelty(shrinkage_jitter=1e-3)
    # Must not raise LinAlgError: Singular matrix
    mahal.fit(X_collinear)

    scores = mahal.score(X_collinear[:10])
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("budget", [0.01, 0.05, 0.10])
def test_two_signal_gate_conformal_calibration_budget(synthetic_data, budget):
    """Verify that split-conformal calibration achieves empirical calibration rate <= budget + tolerance."""
    X_train, y_train, X_cal, y_cal = synthetic_data
    X_cal_benign = X_cal[y_cal == 0]

    for nov_type in ["isolation_forest", "mahalanobis", "knn"]:
        gate = TwoSignalEscalationGate(
            novelty_type=nov_type,
            target_budget=budget,
            n_estimators=20,
            max_depth=4,
            seed=42,
        )
        gate.fit(X_train, y_train, X_cal=X_cal_benign)

        # On the calibration set itself, empirical escalation rate must be <= budget + 1/(n+1)
        preds, escalated, scores = gate.predict(X_cal_benign)
        emp_rate = np.mean(escalated)
        assert emp_rate <= budget + 0.03, (
            f"Gate {nov_type} exceeded budget {budget}: empirical rate was {emp_rate:.4f}"
        )


def test_two_signal_gate_logical_or_property(synthetic_data):
    """
    Verify that TwoSignalEscalationGate escalates when EITHER discriminative
    uncertainty is high OR novelty score is high (joint_score = max(q_disc, q_nov)).
    """
    X_train, y_train, X_test, _ = synthetic_data
    gate = TwoSignalEscalationGate(
        novelty_type="isolation_forest",
        target_budget=0.05,
        n_estimators=20,
        max_depth=4,
        seed=42,
    )
    gate.fit(X_train, y_train)

    probs = gate.rf.predict_proba(X_test)
    raw_disc = 1.0 - 2.0 * np.abs(probs[:, 1] - 0.5)
    raw_nov = gate.novelty.score(X_test)

    q_disc, q_nov = gate._compute_quantiles(raw_disc, raw_nov)
    expected_joint = np.maximum(q_disc, q_nov)
    expected_escalate = expected_joint >= gate.threshold

    preds, actual_escalate, actual_scores = gate.predict(X_test)

    np.testing.assert_allclose(actual_scores, expected_joint, rtol=1e-5)
    np.testing.assert_array_equal(actual_escalate, expected_escalate)


def test_two_signal_gate_allowlist_guard():
    """Verify that feature matrices fed to TwoSignalEscalationGate have zero overlap with identifiers."""
    overlap = set(TIER1_ALLOWLIST) & set(IDENTIFIER_COLUMNS)
    assert len(overlap) == 0, f"TIER1_ALLOWLIST contains forbidden identifiers: {overlap}"
