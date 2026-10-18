"""
src/gateway/tier2_worker.py
───────────────────────────
Asynchronous Tier 2 Deep Verification Worker for Gateway Ingress.
Consumes requests from escalation_queue, extracts full-text n-gram representations,
and delivers verdicts back to waiting HTTP client connections.
"""

import asyncio
import sqlite3
import time
from typing import Dict

from tier2.track_a_http import extract_raw_http_text


def log_tier2_verdict(path: str, verdict: str, latency_ms: float):
    try:
        conn = sqlite3.connect("gateway_logs.db")
        c = conn.cursor()
        c.execute(
            "CREATE TABLE IF NOT EXISTS tier2_logs (timestamp TEXT, path TEXT, verdict TEXT, latency_ms REAL)"
        )
        c.execute(
            "INSERT INTO tier2_logs VALUES (datetime('now'), ?, ?, ?)",
            (path, verdict, latency_ms),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


async def tier2_worker_loop(escalation_queue: asyncio.Queue):
    """
    Dedicated background worker loop processing escalated traffic.
    """
    from src.gateway.middleware import get_verifier, inflight_events, tier2_verdicts

    print("[Tier2Worker] Worker loop started. Listening for escalated payloads...")

    while True:
        try:
            item = await escalation_queue.get()
            t0 = time.perf_counter()

            req_id = item.get("req_id")
            row = item.get("row")

            verifier = get_verifier()
            raw_text = extract_raw_http_text(row)
            X2 = verifier.vectorizer.transform([raw_text]).toarray()
            pred = int(verifier.tier2_model.predict(X2)[0])

            latency_ms = (time.perf_counter() - t0) * 1000.0
            verdict_str = "MALICIOUS" if pred == 1 else "BENIGN"
            log_tier2_verdict(row.get("URL", ""), verdict_str, latency_ms)

            # Signal waiting client connection
            if req_id:
                tier2_verdicts[req_id] = pred
                event = inflight_events.get(req_id)
                if event:
                    event.set()

            escalation_queue.task_done()

        except Exception as e:
            await asyncio.sleep(0.01)
