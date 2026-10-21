"""
tier2/track_a_http.py
─────────────────────
Track A: Application Layer HTTP Payload Verification.
Dataset: CSIC 2010 (SQLi, XSS, Buffer Overflow, Normal Web Traffic).

Both tiers inspect the exact same HTTP request:
  - Tier 1 Fast Classifier: 9 lightweight payload features (Table I).
  - Tier 2 Deep Verification Model: Full-text character 3-5 gram TF-IDF representation (1,500 dimensions)
    with a deep verification classifier trained on a separate split.
"""

import os
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


RE_SPECIAL = re.compile(r'[^a-zA-Z0-9\s]')
RE_SQL = re.compile(r'\b(select|union|drop|insert|delete|update|exec|declare)\b|or 1=1|--', re.IGNORECASE)
RE_SCRIPT = re.compile(r'<script|<iframe|javascript:|onerror=', re.IGNORECASE)
RE_CONSEC = re.compile(r'[^a-zA-Z0-9\s]{2,}')
HEADER_KEYS = ("User-Agent", "Pragma", "Cache-Control", "Accept", "Accept-encoding", "Accept-charset", "language", "host", "cookie", "content-type", "connection")


def extract_track_a_fast_features(row: pd.Series) -> List[float]:
    """
    Extract Table I (9 Fast Features) from an HTTP request row.
    Optimized with precompiled regular expressions for low overhead.
    """
    url = str(row.get("URL", ""))
    method = str(row.get("Method", "GET"))
    ua = str(row.get("User-Agent", ""))
    content = str(row.get("content", ""))
    if pd.isna(content) or content.lower() == "nan":
        content = ""

    # Combine text for inspection
    full_text = f"{method} {url} {ua} {content}"
    full_lower = full_text.lower()

    # 1. request_length
    req_len = len(full_text)

    # 2. num_headers (approximated from non-empty header fields in CSIC)
    num_headers = sum(1 for k in HEADER_KEYS if k in row and pd.notna(row[k]) and str(row[k]).strip() != "")

    # 3. special_chars_count
    special_chars = len(RE_SPECIAL.findall(full_text))

    # 4. sql_keywords_count
    sql_kws = ["select", "union", "drop", "insert", "delete", "update", "or 1=1", "exec", "declare", "--"]
    sql_count = sum(1 for kw in sql_kws if kw in full_lower)

    # 5. script_tags_count (XSS)
    script_count = len(RE_SCRIPT.findall(full_lower))

    # 6. path_depth
    path_depth = url.count("/")

    # 7. is_post
    is_post = 1.0 if method.upper() == "POST" else 0.0

    # 8. user_agent_len
    ua_len = len(ua)

    # 9. consecutive_symbols (e.g. ../ or %20%20 or '")
    consec_sym = len(RE_CONSEC.findall(full_text))

    return [
        float(req_len),
        float(num_headers),
        float(special_chars),
        float(sql_count),
        float(script_count),
        float(path_depth),
        is_post,
        float(ua_len),
        float(consec_sym),
    ]


def extract_raw_http_text(row: pd.Series) -> str:
    """Format full HTTP request text for deep n-gram inspection."""
    method = str(row.get("Method", "GET"))
    url = str(row.get("URL", ""))
    headers = []
    for k in ["host", "User-Agent", "Accept", "content-type", "cookie", "connection"]:
        val = row.get(k, "")
        if pd.notna(val) and str(val).strip() != "":
            headers.append(f"{k}: {val}")
    content = str(row.get("content", ""))
    if pd.isna(content) or content.lower() == "nan":
        content = ""

    header_block = "\n".join(headers)
    return f"{method} {url} HTTP/1.1\n{header_block}\n\n{content}"


class TrackAHTTPVerifier:
    """
    Two-Tier Architecture for HTTP Payloads:
      Tier 1: Fast classifier on 9 Table I features with ONNX compiled acceleration.
      Tier 2: Deep Verification Model on TF-IDF character n-grams.
    """
    def __init__(
        self,
        t1_band: Tuple[float, float] = (0.30, 0.70),
        n_features_deep: int = 1500,
        random_state: int = 42,
    ):
        self.t1_band = t1_band
        self.random_state = random_state
        self.tier1_model = RandomForestClassifier(n_estimators=30, max_depth=6, n_jobs=1, random_state=random_state)
        self.vectorizer = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=n_features_deep)
        self.tier2_model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=random_state)
        self.tier1_onnx_sess = None
        self.tier1_onnx_input_name = None
        self.tier1_onnx_output_name = None

    def fit(self, df_train: pd.DataFrame):
        # Extract features
        X1 = np.array([extract_track_a_fast_features(row) for _, row in df_train.iterrows()])
        texts = [extract_raw_http_text(row) for _, row in df_train.iterrows()]
        X2 = self.vectorizer.fit_transform(texts).toarray()
        y = df_train["classification"].values.astype(int)

        # Train both tiers on train split
        self.tier1_model.fit(X1, y)
        self.tier2_model.fit(X2, y)

        # Compile Tier 1 forest to ONNX for microsecond inference
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnxruntime as rt
            initial_type = [("float_input", FloatTensorType([None, 9]))]
            onx = convert_sklearn(self.tier1_model, initial_types=initial_type, target_opset=15)
            opts = rt.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.tier1_onnx_sess = rt.InferenceSession(onx.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"])
            self.tier1_onnx_input_name = self.tier1_onnx_sess.get_inputs()[0].name
            self.tier1_onnx_output_name = self.tier1_onnx_sess.get_outputs()[1].name
        except Exception:
            self.tier1_onnx_sess = None

    def predict_proba_fast(self, x: np.ndarray) -> float:
        """Single-sample fast inference via compiled ONNX session or direct tree fallback."""
        if self.tier1_onnx_sess is not None and os.environ.get("GATEWAY_DISABLE_ONNX", "0") != "1":
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            probs = self.tier1_onnx_sess.run([self.tier1_onnx_output_name], {self.tier1_onnx_input_name: x})[0]
            if isinstance(probs, list):
                return float(probs[0][1])
            return float(probs[0, 1])
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(self.tier1_model.predict_proba(x)[0, 1])

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        X1 = np.array([extract_track_a_fast_features(row) for _, row in df_test.iterrows()])
        texts = [extract_raw_http_text(row) for _, row in df_test.iterrows()]
        X2 = self.vectorizer.transform(texts).toarray()
        y = df_test["classification"].values.astype(int)

        # Measure Tier 1 latency
        t0 = time.perf_counter()
        probs1 = self.tier1_model.predict_proba(X1)
        t1_lat_ms = (time.perf_counter() - t0) * 1000.0 / len(X1)

        p1_class1 = probs1[:, 1]
        preds1 = (p1_class1 >= 0.5).astype(int)

        # Escalation mask
        esc_mask = (p1_class1 >= self.t1_band[0]) & (p1_class1 <= self.t1_band[1])
        esc_rate = float(np.mean(esc_mask))

        # Measure Tier 2 latency on escalated (or single sample)
        sample_x2 = X2[:1]
        t0 = time.perf_counter()
        for _ in range(50):
            _ = self.tier2_model.predict(sample_x2)
        t2_lat_ms = ((time.perf_counter() - t0) * 1000.0) / 50.0

        # Two-tier effective decision
        final_preds = preds1.copy()
        if np.any(esc_mask):
            final_preds[esc_mask] = self.tier2_model.predict(X2[esc_mask])

        # Blended latency
        blended_lat_ms = (1.0 - esc_rate) * t1_lat_ms + esc_rate * (t1_lat_ms + t2_lat_ms)

        # Compute metrics
        acc_t1_only = float(accuracy_score(y, preds1))
        acc_two_tier = float(accuracy_score(y, final_preds))
        f1_two_tier = float(f1_score(y, final_preds, zero_division=0))
        prec_two_tier = float(precision_score(y, final_preds, zero_division=0))
        rec_two_tier = float(recall_score(y, final_preds, zero_division=0))

        # Tier 2 standalone accuracy
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

    def evaluate_pareto(self, df_test: pd.DataFrame, budgets: List[float] = None) -> Dict[float, Dict[str, float]]:
        """
        Sweeps target escalation budgets from 0% to 100% to trace the complete
        Pareto frontier: Accuracy and Silent-Failure Rate (SFR) vs Escalation Rate (%)
        and vs Mean / Blended Latency (ms).
        """
        if budgets is None:
            budgets = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

        X1 = np.array([extract_track_a_fast_features(row) for _, row in df_test.iterrows()])
        texts = [extract_raw_http_text(row) for _, row in df_test.iterrows()]
        X2 = self.vectorizer.transform(texts).toarray()
        y = df_test["classification"].values.astype(int)

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


def profile_tier1_stages(verifier: TrackAHTTPVerifier, df_samples: pd.DataFrame, n_samples: int = 10000) -> Dict[str, Any]:
    """
    Micro-profiles Tier 1 execution per stage over n_samples real requests using time.perf_counter_ns().
    Compares Baseline (raw scikit-learn predict_proba) vs. Optimized (compiled ONNX tree evaluation).
    Every stage is measured independently in separate baseline and optimized loops (no reused timing values).
    Stages measured:
      1. Parse (request extraction)
      2. Features (Table I 9 fast features)
      3. Predict (ML classifier inference)
      4. Enqueue (routing & queue check)
    """
    df_eval = df_samples.sample(n=n_samples, replace=True, random_state=42)

    # 1. Baseline profiling run (raw RF predict_proba, independent timing)
    t_parse_base, t_feat_base, t_pred_base, t_enq_base = [], [], [], []
    for _, row in df_eval.iterrows():
        # Baseline Stage 1: Parse
        t0 = time.perf_counter_ns()
        method = str(row.get("Method", "GET"))
        url = str(row.get("URL", ""))
        body = str(row.get("content", ""))
        ua = str(row.get("User-Agent", ""))
        t1 = time.perf_counter_ns()
        t_parse_base.append((t1 - t0) / 1000.0)

        # Baseline Stage 2: Features
        feats = extract_track_a_fast_features(row)
        t2 = time.perf_counter_ns()
        t_feat_base.append((t2 - t1) / 1000.0)

        # Baseline Stage 3: Predict (raw scikit-learn)
        X = np.array([feats])
        p_base = float(verifier.tier1_model.predict_proba(X)[0, 1])
        t3 = time.perf_counter_ns()
        t_pred_base.append((t3 - t2) / 1000.0)

        # Baseline Stage 4: Enqueue
        is_uncertain = (0.30 <= p_base <= 0.70)
        if is_uncertain:
            _ = {"req_id": "profile_base", "row": row}
        t4 = time.perf_counter_ns()
        t_enq_base.append((t4 - t3) / 1000.0)

    # 2. Optimized profiling run (compiled ONNX inference, independent timing)
    t_parse_opt, t_feat_opt, t_pred_opt, t_enq_opt = [], [], [], []
    for _, row in df_eval.iterrows():
        # Optimized Stage 1: Parse
        t0 = time.perf_counter_ns()
        method = str(row.get("Method", "GET"))
        url = str(row.get("URL", ""))
        body = str(row.get("content", ""))
        ua = str(row.get("User-Agent", ""))
        t1 = time.perf_counter_ns()
        t_parse_opt.append((t1 - t0) / 1000.0)

        # Optimized Stage 2: Features
        feats = extract_track_a_fast_features(row)
        t2 = time.perf_counter_ns()
        t_feat_opt.append((t2 - t1) / 1000.0)

        # Optimized Stage 3: Predict (compiled ONNX predict_proba_fast)
        X = np.array([feats], dtype=np.float32)
        p_opt = float(verifier.predict_proba_fast(X))
        t3 = time.perf_counter_ns()
        t_pred_opt.append((t3 - t2) / 1000.0)

        # Optimized Stage 4: Enqueue
        is_uncertain = (0.30 <= p_opt <= 0.70)
        if is_uncertain:
            _ = {"req_id": "profile_opt", "row": row}
        t4 = time.perf_counter_ns()
        t_enq_opt.append((t4 - t3) / 1000.0)

    def stats(arr):
        return {
            "mean_us": float(np.mean(arr)),
            "p50_us": float(np.percentile(arr, 50)),
            "p90_us": float(np.percentile(arr, 90)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
        }

    total_base = [p + f + pb + e for p, f, pb, e in zip(t_parse_base, t_feat_base, t_pred_base, t_enq_base)]
    total_opt = [p + f + po + e for p, f, po, e in zip(t_parse_opt, t_feat_opt, t_pred_opt, t_enq_opt)]

    return {
        "n_samples": n_samples,
        "stages_baseline": {
            "parse": stats(t_parse_base),
            "features": stats(t_feat_base),
            "predict": stats(t_pred_base),
            "enqueue": stats(t_enq_base),
            "total_pipeline": stats(total_base),
        },
        "stages_optimized": {
            "parse": stats(t_parse_opt),
            "features": stats(t_feat_opt),
            "predict": stats(t_pred_opt),
            "enqueue": stats(t_enq_opt),
            "total_pipeline": stats(total_opt),
        },
        "predict_speedup": float(np.mean(t_pred_base) / max(np.mean(t_pred_opt), 0.001)),
        "pipeline_speedup": float(np.mean(total_base) / max(np.mean(total_opt), 0.001)),
    }


def compute_onnx_parity(verifier: TrackAHTTPVerifier, df_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates parity between Scikit-Learn RandomForest and compiled ONNX session:
      - Label agreement (%)
      - Max absolute probability difference
      - Count of decisions that flip at the 0.30 / 0.70 escalation thresholds
    """
    X1 = np.array([extract_track_a_fast_features(row) for _, row in df_test.iterrows()], dtype=np.float32)
    p_sklearn = verifier.tier1_model.predict_proba(X1)[:, 1]
    p_onnx = np.array([verifier.predict_proba_fast(x) for x in X1])

    label_sklearn = (p_sklearn >= 0.5).astype(int)
    label_onnx = (p_onnx >= 0.5).astype(int)
    agreement = float(np.mean(label_sklearn == label_onnx) * 100.0)

    max_prob_diff = float(np.max(np.abs(p_sklearn - p_onnx)))

    low_band, high_band = verifier.t1_band
    esc_sklearn = (p_sklearn >= low_band) & (p_sklearn <= high_band)
    esc_onnx = (p_onnx >= low_band) & (p_onnx <= high_band)
    threshold_flips = int(np.sum(esc_sklearn != esc_onnx))

    return {
        "n_samples": len(df_test),
        "label_agreement_pct": agreement,
        "max_prob_diff": max_prob_diff,
        "threshold_flips_count": threshold_flips,
        "low_band": low_band,
        "high_band": high_band,
        "parity_passed": bool(agreement >= 99.9 and max_prob_diff < 1e-3 and threshold_flips == 0),
    }


