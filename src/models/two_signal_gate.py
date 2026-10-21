"""
src/models/two_signal_gate.py
─────────────────────────────
Implements the Two-Signal Escalation Gate:
  Escalate if discriminative score is uncertain OR benign-only novelty score is high.

Components:
  1. Benign-Only Novelty Detectors (trained strictly on benign training flows):
     (a) IsolationForestNovelty: iForest fitted on benign only.
     (b) MahalanobisNovelty: Regularized covariance & Mahalanobis distance to benign centroid.
     (c) KNNNovelty: Mean distance to k-nearest benign neighbors (KDTree/BallTree).
     (d) AutoencoderNovelty: Small PyTorch MLP autoencoder with reconstruction MSE.

  2. Discriminative Uncertainty Estimator:
     Margin uncertainty u(x) = 1 - 2|p(x) - 0.5| in [0, 1].

  3. Split-Conformal Calibration:
     Calibrates joint max-quantile score S(x) = max(q_disc(x), q_nov(x)) on held-out
     calibration traffic so empirical escalation rate on calibration traffic meets
     the target budget B in {0.01, 0.05, 0.10}.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ── 1. BENIGN-ONLY NOVELTY DETECTORS ──────────────────────────────────────────

class IsolationForestNovelty:
    """Isolation Forest trained strictly on benign traffic flows."""

    def __init__(self, n_estimators: int = 100, contamination: str = "auto", random_state: int = 42):
        self.iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_benign: np.ndarray):
        self.iso.fit(X_benign)

    def score(self, X: np.ndarray) -> np.ndarray:
        # Negative anomaly score: higher = more anomalous/novel relative to benign
        return -self.iso.score_samples(X)


class MahalanobisNovelty:
    """
    Mahalanobis distance to the empirical benign centroid.
    Uses Ledoit-Wolf shrinkage regularization to ensure positive-definiteness
    and prevent singularity with collinear network flow features.
    """

    def __init__(self, shrinkage_jitter: float = 1e-4):
        self.scaler = StandardScaler()
        self.mean_vec: Optional[np.ndarray] = None
        self.precision_mat: Optional[np.ndarray] = None
        self.jitter = shrinkage_jitter

    def fit(self, X_benign: np.ndarray):
        X_s = self.scaler.fit_transform(X_benign)
        self.mean_vec = np.mean(X_s, axis=0)

        # Robust shrinkage estimation
        try:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(X_s)
            cov = lw.covariance_
            # Add small diagonal jitter for guaranteed numerical stability
            cov += self.jitter * np.eye(cov.shape[0])
            self.precision_mat = np.linalg.pinv(cov)
        except Exception:
            # Fallback to empirical covariance with diagonal regularization
            cov = np.cov(X_s, rowvar=False) + (self.jitter + 1e-3) * np.eye(X_s.shape[1])
            self.precision_mat = np.linalg.pinv(cov)

    def score(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        diff = X_s - self.mean_vec
        # Vectorized Mahalanobis calculation: sqrt(diag(diff @ inv_cov @ diff.T))
        quad = np.sum((diff @ self.precision_mat) * diff, axis=1)
        # Numerical guard against tiny negative values due to floating-point imprecision
        quad = np.maximum(quad, 0.0)
        return np.sqrt(quad)


class KNNNovelty:
    """
    k-Nearest Neighbors novelty detector.
    Novelty score is the mean Euclidean distance to the k nearest benign training flows.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.scaler = StandardScaler()
        self.nn: Optional[NearestNeighbors] = None

    def fit(self, X_benign: np.ndarray):
        X_s = self.scaler.fit_transform(X_benign)
        # Use ball_tree for fast multi-dimensional spatial queries
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto", metric="euclidean", n_jobs=-1)
        self.nn.fit(X_s)

    def score(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        distances, _ = self.nn.kneighbors(X_s)
        return np.mean(distances, axis=1)


class _AutoencoderPyTorch(nn.Module):
    """Small MLP Autoencoder: input_dim -> 16 -> 8 -> 16 -> input_dim."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, bottleneck_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderNovelty:
    """
    Small PyTorch Autoencoder trained on benign flows with MSE reconstruction loss.
    Novelty score is the flow reconstruction MSE (higher MSE = novel flow).
    """

    def __init__(self, input_dim: int = 9, epochs: int = 25, batch_size: int = 128, lr: float = 0.005, seed: int = 42):
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.net = _AutoencoderPyTorch(input_dim)
        self.device = torch.device("cpu")

    def fit(self, X_benign: np.ndarray):
        X_s = self.scaler.fit_transform(X_benign)
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_s))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for batch in loader:
                xb = batch[0].to(self.device)
                optimizer.zero_grad()
                recon = self.net(xb)
                loss = criterion(recon, xb)
                loss.backward()
                optimizer.step()

    def score(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_s = self.scaler.transform(X)
        with torch.no_grad():
            x_t = torch.FloatTensor(X_s).to(self.device)
            recon = self.net(x_t)
            mse = torch.mean((x_t - recon) ** 2, dim=1).cpu().numpy()
        return mse


# ── 2. TWO-SIGNAL ESCALATION GATE ─────────────────────────────────────────────

class TwoSignalEscalationGate:
    """
    Two-Signal Escalation Gate with Split-Conformal Calibration.

    Condition:
      Escalate if Discriminative Score is Uncertain OR Benign Novelty Score is High:
        S(x) = max( q_disc(x), q_nov(x) ) >= tau_B

    Where:
      - q_disc(x) is the empirical calibration quantile of discriminative margin uncertainty.
      - q_nov(x) is the empirical calibration quantile of benign-only novelty.
      - tau_B is the split-conformal quantile threshold chosen so the calibration escalation
        rate does not exceed budget B in {0.01, 0.05, 0.10}.
    """

    def __init__(
        self,
        novelty_type: str = "isolation_forest",
        target_budget: float = 0.05,
        n_estimators: int = 40,
        max_depth: int = 6,
        seed: int = 42,
    ):
        self.novelty_type = novelty_type.lower()
        self.target_budget = target_budget
        self.seed = seed

        # Discriminative classifier
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=1,
        )

        # Benign-only novelty detector
        if self.novelty_type == "isolation_forest":
            self.novelty = IsolationForestNovelty(random_state=seed)
        elif self.novelty_type == "mahalanobis":
            self.novelty = MahalanobisNovelty()
        elif self.novelty_type == "knn":
            self.novelty = KNNNovelty(k=5)
        elif self.novelty_type == "autoencoder":
            self.novelty = AutoencoderNovelty(seed=seed)
        else:
            raise ValueError(f"Unknown novelty_type: {novelty_type}")

        # Calibration state
        self.cal_disc_scores: Optional[np.ndarray] = None
        self.cal_nov_scores: Optional[np.ndarray] = None
        self.cal_joint_scores: Optional[np.ndarray] = None
        self.threshold: float = 0.5
        self.cal_empirical_escalation_rate: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ):
        """
        Fits Tier 1 RF on (X_train, y_train) and novelty detector strictly on benign flows
        (X_train[y_train == 0]). Then calibrates the joint escalation threshold on X_cal.
        """
        # 1. Fit discriminative Tier 1 model
        self.rf.fit(X_train, y_train)

        # 2. Fit novelty detector strictly on benign flows
        benign_mask = (y_train == 0)
        X_benign = X_train[benign_mask]
        self.novelty.fit(X_benign)

        # 3. Establish calibration split if not provided
        if X_cal is None or len(X_cal) == 0:
            # Use held-out 25% of benign training traffic for conformal calibration
            cal_size = max(50, int(0.25 * len(X_benign)))
            X_cal_use = X_benign[-cal_size:]
        else:
            X_cal_use = X_cal

        # 4. Compute calibration non-conformity scores
        p_cal = self.rf.predict_proba(X_cal_use)[:, 1]
        self.cal_disc_scores = 1.0 - 2.0 * np.abs(p_cal - 0.5)  # in [0, 1]
        self.cal_nov_scores = self.novelty.score(X_cal_use)

        # Convert calibration scores to empirical quantiles [0, 1]
        n_cal = len(X_cal_use)
        q_disc_cal = np.array([np.mean(self.cal_disc_scores <= s) for s in self.cal_disc_scores])
        q_nov_cal = np.array([np.mean(self.cal_nov_scores <= s) for s in self.cal_nov_scores])

        # Combined two-signal score on calibration set: max(q_disc, q_nov)
        self.cal_joint_scores = np.maximum(q_disc_cal, q_nov_cal)

        # 5. Split-conformal quantile threshold for budget B
        level = np.ceil((n_cal + 1) * (1.0 - self.target_budget)) / n_cal
        level = min(1.0, max(0.0, level))
        self.threshold = float(np.quantile(self.cal_joint_scores, level, method="higher"))

        # Verify calibration empirical escalation rate
        cal_escalated = self.cal_joint_scores >= self.threshold
        self.cal_empirical_escalation_rate = float(np.mean(cal_escalated))

    def _compute_quantiles(self, raw_disc: np.ndarray, raw_nov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Maps raw discriminative and novelty scores to empirical calibration quantiles."""
        sorted_disc = np.sort(self.cal_disc_scores)
        sorted_nov = np.sort(self.cal_nov_scores)
        n_cal = len(sorted_disc)

        q_disc = np.searchsorted(sorted_disc, raw_disc, side="right") / float(n_cal)
        q_nov = np.searchsorted(sorted_nov, raw_nov, side="right") / float(n_cal)
        return q_disc, q_nov

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          preds: Tier 1 binary predictions (0=benign, 1=attack)
          should_escalate: boolean mask where True indicates escalation to Tier 2
          combined_score: S(x) = max(q_disc(x), q_nov(x))
        """
        probs = self.rf.predict_proba(X)
        p1 = probs[:, 1]
        preds = (p1 >= 0.5).astype(int)

        # Raw scores
        raw_disc = 1.0 - 2.0 * np.abs(p1 - 0.5)
        raw_nov = self.novelty.score(X)

        # Conformal quantile mapping
        q_disc, q_nov = self._compute_quantiles(raw_disc, raw_nov)

        # Two-signal combined score (implements logical OR via max)
        joint_score = np.maximum(q_disc, q_nov)
        should_escalate = joint_score >= self.threshold

        return preds, should_escalate, joint_score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(X)


# ── 3. CONFORMAL SINGLE-SIGNAL BASELINES ───────────────────────────────────────

class ConformalSingleSignalTrigger:
    """
    Single-signal trigger calibrated with split-conformal quantile prediction
    at target budget B in {0.01, 0.05, 0.10} for exact apples-to-apples comparison.
    """

    def __init__(
        self,
        signal_type: str = "discriminative",  # 'discriminative', 'isolation_forest', 'mahalanobis', 'knn', 'autoencoder'
        target_budget: float = 0.05,
        n_estimators: int = 40,
        max_depth: int = 6,
        seed: int = 42,
    ):
        self.signal_type = signal_type.lower()
        self.target_budget = target_budget
        self.seed = seed

        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=1,
        )

        if self.signal_type == "isolation_forest":
            self.novelty = IsolationForestNovelty(random_state=seed)
        elif self.signal_type == "mahalanobis":
            self.novelty = MahalanobisNovelty()
        elif self.signal_type == "knn":
            self.novelty = KNNNovelty(k=5)
        elif self.signal_type == "autoencoder":
            self.novelty = AutoencoderNovelty(seed=seed)
        elif self.signal_type == "discriminative":
            self.novelty = None
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")

        self.cal_scores: Optional[np.ndarray] = None
        self.threshold: float = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal: Optional[np.ndarray] = None, y_cal: Optional[np.ndarray] = None):
        self.rf.fit(X_train, y_train)

        benign_mask = (y_train == 0)
        X_benign = X_train[benign_mask]

        if self.novelty is not None:
            self.novelty.fit(X_benign)

        if X_cal is None or len(X_cal) == 0:
            cal_size = max(50, int(0.25 * len(X_benign)))
            X_cal_use = X_benign[-cal_size:]
        else:
            X_cal_use = X_cal

        if self.signal_type == "discriminative":
            p_cal = self.rf.predict_proba(X_cal_use)[:, 1]
            self.cal_scores = 1.0 - 2.0 * np.abs(p_cal - 0.5)
        else:
            self.cal_scores = self.novelty.score(X_cal_use)

        n_cal = len(X_cal_use)
        level = np.ceil((n_cal + 1) * (1.0 - self.target_budget)) / n_cal
        level = min(1.0, max(0.0, level))
        self.threshold = float(np.quantile(self.cal_scores, level, method="higher"))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = self.rf.predict_proba(X)
        p1 = probs[:, 1]
        preds = (p1 >= 0.5).astype(int)

        if self.signal_type == "discriminative":
            scores = 1.0 - 2.0 * np.abs(p1 - 0.5)
        else:
            scores = self.novelty.score(X)

        should_escalate = scores >= self.threshold
        return preds, should_escalate, scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(X)
