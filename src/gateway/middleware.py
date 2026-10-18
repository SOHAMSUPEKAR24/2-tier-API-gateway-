"""
src/gateway/middleware.py
─────────────────────────
Implements TASK 6: Real Gateway Tier 1 Screening & Async Escalation Queue.

Features:
  - Real Tier 1 classifier using Track A Table I payload features.
  - Real Tier 2 Deep Verification worker communicating via async queue.
  - Configurable escalated-request behavior:
      * 'hold_with_timeout': Holds client HTTP connection up to timeout_ms waiting for Tier 2 verdict.
      * 'async_quarantine': Immediately returns HTTP 202 Accepted ('Quarantined for async verification').
  - Configurable operational inspection mode:
      * 'two_tier': Standard two-tier edge screening + escalation.
      * 'pure_tier2': Monolithic deep verification on 100% of traffic.
      * 'no_inspection': Passthrough without inspection.
  - Real measured latency and queue telemetry written to SQLite gateway_logs.db.
"""

import asyncio
import os
import sqlite3
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse

from tier2.track_a_http import TrackAHTTPVerifier, extract_track_a_fast_features, extract_raw_http_text


# ── Configurable Environment Variables ────────────────────────────────────────
INSPECTION_MODE = os.environ.get("GATEWAY_INSPECTION_MODE", "two_tier")  # 'two_tier', 'pure_tier2', 'no_inspection'
ESCALATION_BEHAVIOR = os.environ.get("GATEWAY_ESCALATION_BEHAVIOR", "hold_with_timeout")  # 'hold_with_timeout', 'async_quarantine'
ESCALATION_TIMEOUT_MS = float(os.environ.get("GATEWAY_TIMEOUT_MS", "60.0"))

# SQLite setup
DB_PATH = "gateway_logs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS access_logs (
            timestamp TEXT,
            method TEXT,
            path TEXT,
            latency_ms REAL,
            decision TEXT,
            escalated INTEGER,
            mode TEXT
        )"""
    )
    conn.commit()
    conn.close()


init_db()

# Async escalation queue and in-flight response futures
escalation_queue: asyncio.Queue = asyncio.Queue()
inflight_events: Dict[str, asyncio.Event] = {}
tier2_verdicts: Dict[str, int] = {}

# Global Real Model
global_verifier: Optional[TrackAHTTPVerifier] = None


def get_verifier() -> TrackAHTTPVerifier:
    global global_verifier
    if global_verifier is None:
        print("[Gateway] Initializing and pre-fitting Real Track A HTTP Verifier...")
        csic_csv = "/Users/sohamsupekar/Desktop/MajorProject/DATASET/csic_database.csv"
        df_all = pd.read_csv(csic_csv)
        df_benign = df_all[df_all["classification"] == 0].head(2000)
        df_attack = df_all[df_all["classification"] == 1].head(2000)
        df_fit = pd.concat([df_benign, df_attack]).sample(frac=1.0, random_state=42)
        v = TrackAHTTPVerifier(t1_band=(0.30, 0.70), random_state=42)
        v.fit(df_fit)
        global_verifier = v
        print("[Gateway] Verifier ready for wire-speed edge screening.")
    return global_verifier


def log_request_sync(method: str, path: str, latency_ms: float, decision: str, escalated: int, mode: str):
    if os.environ.get("GATEWAY_DISABLE_LOGGING", "0") == "1":
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO access_logs VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
            (method, path, latency_ms, decision, escalated, mode),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        pass



async def screen_request(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Core Ingress Proxy Screening Logic (Algorithm 1).
    """
    start_time = time.perf_counter()

    # Extract request properties
    method = request.method
    path = request.url.path
    headers = dict(request.headers)
    body = ""
    try:
        raw_body = await request.body()
        body = raw_body.decode("utf-8", errors="ignore")
    except Exception:
        pass

    req_row = pd.Series({
        "Method": method,
        "URL": f"http://localhost:8000{path}",
        "User-Agent": headers.get("user-agent", ""),
        "content": body,
        "host": headers.get("host", "localhost"),
    })

    # Mode 1: No Inspection Passthrough Baseline
    if INSPECTION_MODE == "no_inspection":
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        background_tasks.add_task(log_request_sync, method, path, latency_ms, "FORWARDED", 0, "no_inspection")
        return JSONResponse({"status": "Forwarded upstream (no inspection)", "latency_ms": latency_ms})

    verifier = get_verifier()

    # Mode 2: Pure Tier 2 (Every request undergoes deep inspection)
    if INSPECTION_MODE == "pure_tier2":
        raw_text = extract_raw_http_text(req_row)
        X2 = verifier.vectorizer.transform([raw_text]).toarray()
        pred = int(verifier.tier2_model.predict(X2)[0])
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = "BLOCKED" if pred == 1 else "FORWARDED"
        background_tasks.add_task(log_request_sync, method, path, latency_ms, decision, 1, "pure_tier2")

        if pred == 1:
            return JSONResponse(status_code=403, content={"detail": "Blocked by Pure Tier 2 Deep Verification", "latency_ms": latency_ms})
        return JSONResponse({"status": "Forwarded upstream (pure tier 2 verified)", "latency_ms": latency_ms})

    # Mode 3: Two-Tier (Algorithm 1)
    # 1. Fast Feature Extraction (<0.1ms)
    t1_feats = np.array([extract_track_a_fast_features(req_row)])
    p_attack = float(verifier.tier1_model.predict_proba(t1_feats)[0, 1])

    # Escalation check
    low_band, high_band = verifier.t1_band
    is_uncertain = (p_attack >= low_band) and (p_attack <= high_band)

    if not is_uncertain:
        # Decided at wire speed by Tier 1
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        if p_attack > high_band:
            background_tasks.add_task(log_request_sync, method, path, latency_ms, "BLOCKED", 0, "two_tier")
            return JSONResponse(status_code=403, content={"detail": "Blocked by Tier 1 Fast Edge Classifier", "latency_ms": latency_ms})

        background_tasks.add_task(log_request_sync, method, path, latency_ms, "FORWARDED", 0, "two_tier")
        return JSONResponse({"status": "Forwarded upstream (Tier 1 confident benign)", "latency_ms": latency_ms})

    # Escalated request
    req_id = f"{time.time()}_{id(request)}"
    event = asyncio.Event()
    inflight_events[req_id] = event

    # Enqueue to background Tier 2 worker
    await escalation_queue.put({
        "req_id": req_id,
        "row": req_row,
    })

    # Handle escalation according to configurable policy
    if ESCALATION_BEHAVIOR == "async_quarantine":
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        background_tasks.add_task(log_request_sync, method, path, latency_ms, "QUARANTINED_ASYNC", 1, "two_tier")
        return JSONResponse(
            status_code=202,
            content={"status": "Quarantined: forwarded to Tier 2 for async verification", "latency_ms": latency_ms},
        )

    # Otherwise: 'hold_with_timeout'
    try:
        timeout_sec = ESCALATION_TIMEOUT_MS / 1000.0
        await asyncio.wait_for(event.wait(), timeout=timeout_sec)
        t2_verdict = tier2_verdicts.pop(req_id, 0)
        inflight_events.pop(req_id, None)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        if t2_verdict == 1:
            background_tasks.add_task(log_request_sync, method, path, latency_ms, "BLOCKED", 1, "two_tier")
            return JSONResponse(status_code=403, content={"detail": "Blocked by Tier 2 Deep Verification", "latency_ms": latency_ms})

        background_tasks.add_task(log_request_sync, method, path, latency_ms, "FORWARDED", 1, "two_tier")
        return JSONResponse({"status": "Forwarded upstream (Tier 2 verified benign)", "latency_ms": latency_ms})

    except asyncio.TimeoutError:
        inflight_events.pop(req_id, None)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        # Fallback to Tier 1 verdict on timeout
        if p_attack >= 0.5:
            background_tasks.add_task(log_request_sync, method, path, latency_ms, "TIMEOUT_BLOCKED", 1, "two_tier")
            return JSONResponse(status_code=403, content={"detail": "Tier 2 Timeout: Blocked by edge fallback", "latency_ms": latency_ms})

        background_tasks.add_task(log_request_sync, method, path, latency_ms, "TIMEOUT_FORWARDED", 1, "two_tier")
        return JSONResponse({"status": "Tier 2 Timeout: Forwarded by edge fallback", "latency_ms": latency_ms})

