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


def log_tier2_verdict(path: str, verdict: str, service_time_ms: float, held: int, wait_time_ms: float):
    try:
        conn = sqlite3.connect("gateway_logs.db")
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS tier2_logs (
                timestamp TEXT,
                path TEXT,
                verdict TEXT,
                service_time_ms REAL,
                held INTEGER,
                wait_time_ms REAL
            )"""
        )
        c.execute(
            "INSERT INTO tier2_logs VALUES (datetime('now'), ?, ?, ?, ?, ?)",
            (path, verdict, service_time_ms, held, wait_time_ms),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def sync_t2_predict(verifier, raw_text: str) -> int:
    X2 = verifier.vectorizer.transform([raw_text]).toarray()
    return int(verifier.tier2_model.predict(X2)[0])


async def tier2_worker_loop(escalation_queue: asyncio.Queue):
    """
    Dedicated background worker loop processing escalated traffic.
    """
    from src.gateway.middleware import get_verifier, inflight_events, tier2_verdicts
    import src.gateway.middleware as mw

    print("[Tier2Worker] Worker loop started. Listening for escalated payloads...")

    while True:
        try:
            item = await escalation_queue.get()
            t0 = time.perf_counter()

            req_id = item.get("req_id")
            row = item.get("row")
            t_inbound = item.get("t_inbound", t0)
            behavior = item.get("behavior", "hold_with_timeout")
            is_held = 1 if behavior != "async_quarantine" else 0

            verifier = get_verifier()
            raw_text = extract_raw_http_text(row)
            pred = await asyncio.to_thread(sync_t2_predict, verifier, raw_text)

            # Align service time to empirical capacity (516 req/s across 8 workers = 15.5 ms/item)
            target_service_sec = float(os.environ.get("GATEWAY_TIER2_SERVICE_TIME_SEC", "0.0155"))
            elapsed = time.perf_counter() - t0
            if elapsed < target_service_sec:
                await asyncio.sleep(target_service_sec - elapsed)

            t1 = time.perf_counter()
            service_time_ms = (t1 - t0) * 1000.0
            wait_time_ms = max(0.0, (t0 - t_inbound) * 1000.0)
            verdict_str = "MALICIOUS" if pred == 1 else "BENIGN"

            # Per-request logging with service time and held/not-held flag
            log_tier2_verdict(row.get("URL", ""), verdict_str, service_time_ms, is_held, wait_time_ms)

            # Telemetry tracking for service times and held/not-held requests
            mw.tier2_service_times.append(service_time_ms)
            if is_held == 1:
                mw.tier2_held_count += 1
            else:
                mw.tier2_not_held_count += 1

            # Check if this request timed out under fail-open (held mode)
            if req_id and req_id in mw.timed_out_fail_open_reqs:
                mw.timed_out_fail_open_reqs.discard(req_id)
                if pred == 1:
                    mw.hold_fail_open_malicious_exposed_count += 1

            # Record exposure window for async quarantine
            if behavior == "async_quarantine":
                exposure_ms = (t1 - t_inbound) * 1000.0
                mw.async_exposure_windows.append(exposure_ms)
                if pred == 1:
                    mw.async_malicious_exposed_count += 1

            # Signal waiting client connection
            if req_id:
                tier2_verdicts[req_id] = pred
                event = inflight_events.get(req_id)
                if event:
                    event.set()

            escalation_queue.task_done()

        except Exception as e:
            await asyncio.sleep(0.01)
