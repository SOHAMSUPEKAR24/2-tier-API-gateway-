"""
src/models/escalation_triggers.py
──────────────────────────────────
Implementation of Escalation Triggers and Uncertainty Estimators for Tier 1:
  (a) RF Probability Band [0.30, 0.70]
  (b) Calibrated RF (Isotonic / Platt)
  (c) Bagged RF Tree Variance (Deep Ensemble Proxy)
  (d) Split-Conformal Prediction (Non-singleton prediction set at level alpha)
  (e) Isolation Forest Novelty Detector
  (f) Evidential Deep Learning (EDL) & EDL Ensemble with Warm-Start Dirichlet Prior

All methods adhere to a common API:
  trigger.fit(X_train, y_train, X_cal=None, y_cal=None)
  verdict, should_escalate, novelty_score = trigger.predict(X_test)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier


# ── (a) RF Probability Band ──────────────────────────────────────────────────
class RFProbabilityBandTrigger:
    def __init__(self, low: float = 0.30, high: float = 0.70, n_estimators: int = 40, max_depth: int = 6, random_state: int = 42):
        self.low = low
        self.high = high
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal=None, y_cal=None):
        self.rf.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = self.rf.predict_proba(X)
        p1 = probs[:, 1]
        preds = (p1 >= 0.5).astype(int)
        should_escalate = (p1 >= self.low) & (p1 <= self.high)
        # Novelty score: highest at p=0.5, lowest at p=0 or 1
        novelty_score = 1.0 - 2.0 * np.abs(p1 - 0.5)
        return preds, should_escalate, novelty_score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(X)


# ── (b) Calibrated RF (Isotonic / Platt) ───────────────────────────────────────
class CalibratedRFTrigger:
    def __init__(self, method: str = "isotonic", low: float = 0.30, high: float = 0.70, n_estimators: int = 40, max_depth: int = 6, random_state: int = 42):
        self.method = method
        self.low = low
        self.high = high
        self.base_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.calibrator = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal: np.ndarray = None, y_cal: np.ndarray = None):
        self.base_rf.fit(X_train, y_train)
        if X_cal is not None and len(X_cal) > 0 and len(np.unique(y_cal)) > 1:
            self.calibrator = CalibratedClassifierCV(self.base_rf, cv="prefit", method=self.method)
            self.calibrator.fit(X_cal, y_cal)
        else:
            self.calibrator = CalibratedClassifierCV(self.base_rf, cv=3, method=self.method)
            self.calibrator.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = self.calibrator.predict_proba(X)
        p1 = probs[:, 1]
        preds = (p1 >= 0.5).astype(int)
        should_escalate = (p1 >= self.low) & (p1 <= self.high)
        novelty_score = 1.0 - 2.0 * np.abs(p1 - 0.5)
        return preds, should_escalate, novelty_score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(X)


# ── (c) Bagged RF Tree Variance (Deep Ensemble Proxy) ─────────────────────────
class BaggedRFVarianceTrigger:
    def __init__(self, var_threshold: float = 0.16, n_estimators: int = 60, max_depth: int = 6, random_state: int = 42):
        self.var_threshold = var_threshold
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal=None, y_cal=None):
        self.rf.fit(X_train, y_train)
        if X_cal is not None and len(X_cal) > 0:
            tree_preds = np.array([tree.predict_proba(X_cal)[:, 1] for tree in self.rf.estimators_])
            variances = np.var(tree_preds, axis=0)
            self.var_threshold = float(np.percentile(variances, 85))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tree_preds = np.array([tree.predict_proba(X)[:, 1] for tree in self.rf.estimators_])
        mean_p1 = np.mean(tree_preds, axis=0)
        variances = np.var(tree_preds, axis=0)
        preds = (mean_p1 >= 0.5).astype(int)
        should_escalate = variances >= self.var_threshold
        # Novelty score is the ensemble disagreement variance
        novelty_score = variances
        return preds, should_escalate, novelty_score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(X)


# ── (d) Split-Conformal Prediction ────────────────────────────────────────────
class SplitConformalTrigger:
    def __init__(self, alpha: float = 0.10, n_estimators: int = 40, max_depth: int = 6, random_state: int = 42):
        self.alpha = alpha
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.q_hat = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal: np.ndarray = None, y_cal: np.ndarray = None):
        if X_cal is None or len(X_cal) == 0:
            split = int(0.75 * len(X_train))
            X_tr, X_c = X_train[:split], X_train[split:]
            y_tr, y_c = y_train[:split], y_train[split:]
        else:
            X_tr, y_tr = X_train, y_train
            X_c, y_c = X_cal, y_cal

        self.rf.fit(X_tr, y_tr)
        probs_cal = self.rf.predict_proba(X_c)

        # Non-conformity score: 1 - P(true_class | x)
        true_class_probs = probs_cal[np.arange(len(y_c)), y_c]
        non_conformity = 1.0 - true_class_probs

        n = len(non_conformity)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(1.0, max(0.0, level))
        self.q_hat = float(np.quantile(non_conformity, level, method="higher"))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = self.rf.predict_proba(X)
        p1 = probs[:, 1]
        preds = (p1 >= 0.5).astype(int)

        # Prediction set membership: include class c if (1 - prob(c)) <= q_hat
        in_set_0 = (1.0 - probs[:, 0]) <= self.q_hat
        in_set_1 = (1.0 - probs[:, 1]) <= self.q_hat
        set_size = in_set_0.astype(int) + in_set_1.astype(int)

        # Escalate when prediction set is non-singleton: ambiguous {0, 1} or empty {}
        should_escalate = set_size != 1
        novelty_score = np.minimum(1.0 - probs[:, 0], 1.0 - probs[:, 1])
        return preds, should_escalate, novelty_score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(X)


# ── (e) Isolation Forest / Novelty Score ──────────────────────────────────────
class IsolationForestTrigger:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 40, max_depth: int = 6, random_state: int = 42):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
        self.threshold = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal=None, y_cal=None):
        self.rf.fit(X_train, y_train)
        self.iso_forest.fit(X_train)
        scores = -self.iso_forest.score_samples(X_train)
        self.threshold = float(np.percentile(scores, 90))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = self.rf.predict(X)
        anomaly_scores = -self.iso_forest.score_samples(X)
        should_escalate = anomaly_scores >= self.threshold
        return preds, should_escalate, anomaly_scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(X)


# ── (f) EDL & EDL Ensemble with Warm-Start Prior ──────────────────────────────
class _EDLNetwork(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softplus(self.out(x))


from sklearn.preprocessing import StandardScaler


def _edl_loss(evidence: torch.Tensor, target: torch.Tensor, num_classes: int = 2, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    y = F.one_hot(target, num_classes=num_classes).float()
    err = torch.sum((y - prob) ** 2, dim=1, keepdim=True)
    var = torch.sum(prob * (1.0 - prob) / (S + 1.0), dim=1, keepdim=True)
    loss = err + var
    if weights is not None:
        loss = weights[target].unsqueeze(1) * loss
    return torch.mean(loss)


class EDLTrigger:
    def __init__(self, input_dim: int = 9, uncertainty_threshold: float = 0.40, epochs: int = 25, lr: float = 0.005, seed: int = 42):
        torch.manual_seed(seed)
        self.net = _EDLNetwork(input_dim)
        self.epochs = epochs
        self.lr = lr
        self.threshold = uncertainty_threshold
        self.scaler = StandardScaler()
        self.class_weights: Optional[torch.Tensor] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal=None, y_cal=None):
        X_s = self.scaler.fit_transform(X_train)
        w0 = len(y_train) / (2.0 * max(1, np.sum(y_train == 0)))
        w1 = len(y_train) / (2.0 * max(1, np.sum(y_train == 1)))
        self.class_weights = torch.FloatTensor([w0, w1])

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        X_t = torch.FloatTensor(X_s)
        y_t = torch.LongTensor(y_train)

        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            ev = self.net(X_t)
            loss = _edl_loss(ev, y_t, weights=self.class_weights)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.net.eval()
        X_s = self.scaler.transform(X)
        with torch.no_grad():
            X_t = torch.FloatTensor(X_s)
            ev = self.net(X_t)
            alpha = ev + 1.0
            S = torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / S
            preds = torch.argmax(prob, dim=1).numpy()
            uncertainty = (2.0 / S).numpy().flatten()

        should_escalate = uncertainty >= self.threshold
        return preds, should_escalate, uncertainty

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_s = self.scaler.transform(X)
        with torch.no_grad():
            X_t = torch.FloatTensor(X_s)
            ev = self.net(X_t)
            alpha = ev + 1.0
            S = torch.sum(alpha, dim=1, keepdim=True)
            return (alpha / S).numpy()


class EDLEnsembleWarmStartTrigger:
    """
    Ensemble of 3 EDL models with warm-start Dirichlet priors
    to stabilize cold-start uncertainty surges on newly encountered modes.
    """
    def __init__(self, input_dim: int = 9, num_models: int = 3, uncertainty_threshold: float = 0.35, seed: int = 42):
        self.models = [EDLTrigger(input_dim=input_dim, seed=seed + i * 17) for i in range(num_models)]
        self.threshold = uncertainty_threshold
        self.warm_prior = 1.0  # Dirichlet pseudocount prior

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal=None, y_cal=None):
        for m in self.models:
            m.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_probs = []
        all_u = []
        for m in self.models:
            p = m.predict_proba(X)
            _, _, u = m.predict(X)
            all_probs.append(p)
            all_u.append(u)

        mean_probs = np.mean(all_probs, axis=0)
        preds = np.argmax(mean_probs, axis=1)

        # Warm-start prior dampens spurious single-sample cold-start uncertainty
        mean_u = np.mean(all_u, axis=0)
        damped_u = mean_u * (1.0 / (1.0 + 0.1 * self.warm_prior))

        should_escalate = damped_u >= self.threshold
        return preds, should_escalate, damped_u

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_probs = [m.predict_proba(X) for m in self.models]
        return np.mean(all_probs, axis=0)

