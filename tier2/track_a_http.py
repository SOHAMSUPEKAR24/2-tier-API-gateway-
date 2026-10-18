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

import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def extract_track_a_fast_features(row: pd.Series) -> List[float]:
    """
    Extract Table I (9 Fast Features) from an HTTP request row.
    """
    url = str(row.get("URL", ""))
    method = str(row.get("Method", "GET"))
    ua = str(row.get("User-Agent", ""))
    content = str(row.get("content", ""))
    if pd.isna(content) or content.lower() == "nan":
        content = ""

    # Combine text for inspection
    full_text = f"{method} {url} {ua} {content}"

    # 1. request_length
    req_len = len(full_text)

    # 2. num_headers (approximated from non-empty header fields in CSIC)
    header_keys = ["User-Agent", "Pragma", "Cache-Control", "Accept", "Accept-encoding", "Accept-charset", "language", "host", "cookie", "content-type", "connection"]
    num_headers = sum(1 for k in header_keys if k in row and pd.notna(row[k]) and str(row[k]).strip() != "")

    # 3. special_chars_count
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', full_text))

    # 4. sql_keywords_count
    sql_kws = ["select", "union", "drop", "insert", "delete", "update", "or 1=1", "exec", "declare", "--"]
    sql_count = sum(1 for kw in sql_kws if kw in full_text.lower())

    # 5. script_tags_count (XSS)
    script_count = len(re.findall(r'<script|<iframe|javascript:|onerror=', full_text.lower()))

    # 6. path_depth
    path_depth = url.count("/")

    # 7. is_post
    is_post = 1 if method.upper() == "POST" else 0

    # 8. user_agent_len
    ua_len = len(ua)

    # 9. consecutive_symbols (e.g. ../ or %20%20 or '")
    consec_sym = len(re.findall(r'[^a-zA-Z0-9\s]{2,}', full_text))

    return [
        float(req_len),
        float(num_headers),
        float(special_chars),
        float(sql_count),
        float(script_count),
        float(path_depth),
        float(is_post),
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
      Tier 1: Fast classifier on 9 Table I features.
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
        self.tier1_model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=random_state)
        self.vectorizer = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=n_features_deep)
        self.tier2_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state)

    def fit(self, df_train: pd.DataFrame):
        # Extract features
        X1 = np.array([extract_track_a_fast_features(row) for _, row in df_train.iterrows()])
        texts = [extract_raw_http_text(row) for _, row in df_train.iterrows()]
        X2 = self.vectorizer.fit_transform(texts).toarray()
        y = df_train["classification"].values.astype(int)

        # Train both tiers on train split
        self.tier1_model.fit(X1, y)
        self.tier2_model.fit(X2, y)

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
