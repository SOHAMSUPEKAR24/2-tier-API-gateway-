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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse

from tier2.track_a_http import TrackAHTTPVerifier, extract_track_a_fast_features, extract_raw_http_text


import json
from pathlib import Path

# ── Configurable Environment Variables & Shared State ─────────────────────────
INSPECTION_MODE = os.environ.get("GATEWAY_INSPECTION_MODE", "two_tier")  # 'two_tier', 'pure_tier2', 'no_inspection'
ESCALATION_BEHAVIOR = os.environ.get("GATEWAY_ESCALATION_BEHAVIOR", "hold_with_timeout")  # 'hold_with_timeout', 'async_quarantine'
ESCALATION_TIMEOUT_MS = float(os.environ.get("GATEWAY_TIMEOUT_MS", "60.0"))
TIMEOUT_ACTION = os.environ.get("GATEWAY_TIMEOUT_ACTION", "fail_open")  # 'fail_open', 'fail_closed'

STATE_FILE = Path("load_test/run/gateway_state.json")
_state_cache: Dict[str, Any] = {
    "mode": INSPECTION_MODE,
    "behavior": ESCALATION_BEHAVIOR,
    "timeout_ms": ESCALATION_TIMEOUT_MS,
    "timeout_action": TIMEOUT_ACTION,
    "reset_epoch": 0.0,
}
_state_cache_mtime: float = 0.0
_last_reset_epoch: float = 0.0

def load_gateway_state() -> Dict[str, Any]:
    global _state_cache, _state_cache_mtime, _last_reset_epoch
    global total_requests, escalated_requests, timeout_count, fail_open_count, fail_closed_504_count
    global async_total_quarantined, async_malicious_exposed_count, async_exposure_windows
    global INSPECTION_MODE, ESCALATION_BEHAVIOR, ESCALATION_TIMEOUT_MS, TIMEOUT_ACTION
    if STATE_FILE.exists():
        try:
            mtime = os.path.getmtime(STATE_FILE)
            if mtime != _state_cache_mtime:
                with open(STATE_FILE) as f:
                    _state_cache = json.load(f)
                _state_cache_mtime = mtime
                r_epoch = _state_cache.get("reset_epoch", 0.0)
                if r_epoch > _last_reset_epoch:
                    _last_reset_epoch = r_epoch
                    while not escalation_queue.empty():
                        try:
                            escalation_queue.get_nowait()
                            escalation_queue.task_done()
                        except Exception:
                            break
                    inflight_events.clear()
                    tier2_verdicts.clear()
                    reset_telemetry_stats()
            INSPECTION_MODE = _state_cache.get("mode", INSPECTION_MODE)
            ESCALATION_BEHAVIOR = _state_cache.get("behavior", ESCALATION_BEHAVIOR)
            ESCALATION_TIMEOUT_MS = float(_state_cache.get("timeout_ms", ESCALATION_TIMEOUT_MS))
            TIMEOUT_ACTION = _state_cache.get("timeout_action", TIMEOUT_ACTION)
        except Exception:
            pass
    else:
        _state_cache["mode"] = INSPECTION_MODE
        _state_cache["behavior"] = ESCALATION_BEHAVIOR
        _state_cache["timeout_ms"] = ESCALATION_TIMEOUT_MS
        _state_cache["timeout_action"] = TIMEOUT_ACTION
    return _state_cache

def save_gateway_state(updates: Dict[str, Any]):
    global INSPECTION_MODE, ESCALATION_BEHAVIOR, ESCALATION_TIMEOUT_MS, TIMEOUT_ACTION
    st = load_gateway_state().copy()
    st.update(updates)
    if "mode" in updates:
        INSPECTION_MODE = updates["mode"]
    if "behavior" in updates:
        ESCALATION_BEHAVIOR = updates["behavior"]
    if "timeout_ms" in updates:
        ESCALATION_TIMEOUT_MS = float(updates["timeout_ms"])
    if "timeout_action" in updates:
        TIMEOUT_ACTION = updates["timeout_action"]
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = STATE_FILE.with_suffix(f".tmp_{os.getpid()}_{time.time()}")
    with open(temp_file, "w") as f:
        json.dump(st, f)
    os.replace(temp_file, STATE_FILE)
    load_gateway_state()

# SQLite setup
DB_PATH = "gateway_logs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA busy_timeout=5000;")
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
        """CREATE TABLE IF NOT EXISTS request_audit_logs (
            request_id TEXT PRIMARY KEY,
            timestamp TEXT,
            ground_truth INTEGER,
            tier1_decision TEXT,
            path TEXT,
            tier2_verdict TEXT,
            service_time_ms REAL,
            status_code INTEGER,
            reached_origin INTEGER,
            malicious_to_origin INTEGER,
            timed_out INTEGER,
            fail_open INTEGER,
            fail_closed_504 INTEGER,
            run_tag TEXT
        )"""
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_audit_run_tag ON request_audit_logs (run_tag);")
    conn.commit()
    conn.close()


init_db()

# Async escalation queue and in-flight response futures
escalation_queue: asyncio.Queue = asyncio.Queue()
inflight_events: Dict[str, asyncio.Event] = {}
tier2_verdicts: Dict[str, int] = {}

# Telemetry counters for escalation modes and timeouts
total_requests: int = 0
escalated_requests: int = 0
timeout_count: int = 0
fail_open_count: int = 0
fail_closed_504_count: int = 0
async_total_quarantined: int = 0
async_malicious_exposed_count: int = 0
async_exposure_windows: list = []
latency_sum_fast: float = 0.0
latency_count_fast: int = 0
latency_sum_esc: float = 0.0
latency_count_esc: int = 0
timed_out_fail_open_reqs: set = set()
hold_fail_open_malicious_exposed_count: int = 0
tier2_service_times: list = []
tier2_held_count: int = 0
tier2_not_held_count: int = 0


def reset_telemetry_stats():
    global total_requests, escalated_requests, timeout_count, fail_open_count, fail_closed_504_count
    global async_total_quarantined, async_malicious_exposed_count, async_exposure_windows
    global latency_sum_fast, latency_count_fast, latency_sum_esc, latency_count_esc
    global timed_out_fail_open_reqs, hold_fail_open_malicious_exposed_count
    global tier2_service_times, tier2_held_count, tier2_not_held_count
    total_requests = 0
    escalated_requests = 0
    timeout_count = 0
    fail_open_count = 0
    fail_closed_504_count = 0
    async_total_quarantined = 0
    async_malicious_exposed_count = 0
    async_exposure_windows = []
    latency_sum_fast = 0.0
    latency_count_fast = 0
    latency_sum_esc = 0.0
    latency_count_esc = 0
    timed_out_fail_open_reqs = set()
    hold_fail_open_malicious_exposed_count = 0
    tier2_service_times = []
    tier2_held_count = 0
    tier2_not_held_count = 0
    stats_dir = Path("load_test/run/worker_stats")
    if stats_dir.exists():
        for sf in stats_dir.glob("stats_*.json"):
            try:
                sf.unlink()
            except Exception:
                pass


def save_worker_stats():
    stats_dir = Path("load_test/run/worker_stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / f"stats_{os.getpid()}.json"
    with open(stats_file, "w") as f:
        json.dump({
            "pid": os.getpid(),
            "total_requests": total_requests,
            "escalated_requests": escalated_requests,
            "timeout_count": timeout_count,
            "fail_open_count": fail_open_count,
            "fail_closed_504_count": fail_closed_504_count,
            "async_total_quarantined": async_total_quarantined,
            "async_malicious_exposed_count": async_malicious_exposed_count,
            "async_exposure_windows": async_exposure_windows[-500:],
            "latency_sum_fast": latency_sum_fast,
            "latency_count_fast": latency_count_fast,
            "latency_sum_esc": latency_sum_esc,
            "latency_count_esc": latency_count_esc,
            "hold_fail_open_malicious_exposed_count": hold_fail_open_malicious_exposed_count,
            "tier2_held_count": tier2_held_count,
            "tier2_not_held_count": tier2_not_held_count,
            "tier2_service_times": tier2_service_times[-500:],
        }, f)


def get_telemetry_stats() -> Dict[str, Any]:
    save_worker_stats()
    stats_dir = Path("load_test/run/worker_stats")
    tot_req = 0
    esc_req = 0
    t_cnt = 0
    fo_cnt = 0
    fc_cnt = 0
    as_tot = 0
    as_mal = 0
    all_exp = []
    h_mal = 0
    t2_held = 0
    t2_not_held = 0
    all_t2_st = []

    tot_fast_sum = 0.0
    tot_fast_cnt = 0
    tot_esc_sum = 0.0
    tot_esc_cnt = 0

    if stats_dir.exists():
        for sf in stats_dir.glob("stats_*.json"):
            try:
                with open(sf) as f:
                    d = json.load(f)
                tot_req += d.get("total_requests", 0)
                esc_req += d.get("escalated_requests", 0)
                t_cnt += d.get("timeout_count", 0)
                fo_cnt += d.get("fail_open_count", 0)
                fc_cnt += d.get("fail_closed_504_count", 0)
                as_tot += d.get("async_total_quarantined", 0)
                as_mal += d.get("async_malicious_exposed_count", 0)
                all_exp.extend(d.get("async_exposure_windows", []))
                tot_fast_sum += d.get("latency_sum_fast", 0.0)
                tot_fast_cnt += d.get("latency_count_fast", 0)
                tot_esc_sum += d.get("latency_sum_esc", 0.0)
                tot_esc_cnt += d.get("latency_count_esc", 0)
                h_mal += d.get("hold_fail_open_malicious_exposed_count", 0)
                t2_held += d.get("tier2_held_count", 0)
                t2_not_held += d.get("tier2_not_held_count", 0)
                all_t2_st.extend(d.get("tier2_service_times", []))
            except Exception:
                pass

    if not all_exp and async_exposure_windows:
        all_exp = async_exposure_windows
    if not all_t2_st and tier2_service_times:
        all_t2_st = tier2_service_times

    exp_arr = np.array(all_exp) if all_exp else np.array([0.0])
    t2_st_arr = np.array(all_t2_st) if all_t2_st else np.array([0.0])
    mean_f = (tot_fast_sum or latency_sum_fast) / max(tot_fast_cnt or latency_count_fast, 1)
    mean_e = (tot_esc_sum or latency_sum_esc) / max(tot_esc_cnt or latency_count_esc, 1)
    comb_cnt = (tot_fast_cnt or latency_count_fast) + (tot_esc_cnt or latency_count_esc)
    comb_mean = ((tot_fast_sum or latency_sum_fast) + (tot_esc_sum or latency_sum_esc)) / max(comb_cnt, 1)

    tot_malicious_exposed = (as_mal or async_malicious_exposed_count) + (h_mal or hold_fail_open_malicious_exposed_count)

    return {
        "total_requests": tot_req or total_requests,
        "escalated_requests": esc_req or escalated_requests,
        "timeout_count": t_cnt or timeout_count,
        "fail_open_count": fo_cnt or fail_open_count,
        "fail_closed_504_count": fc_cnt or fail_closed_504_count,
        "timeout_rate_pct": float((t_cnt or timeout_count) / max(esc_req or escalated_requests, 1) * 100.0),
        "total_timeout_rate_pct": float((t_cnt or timeout_count) / max(tot_req or total_requests, 1) * 100.0),
        "fail_open_rate_pct": float((fo_cnt or fail_open_count) / max(t_cnt or timeout_count, 1) * 100.0) if (t_cnt or timeout_count) > 0 else 0.0,
        "fail_closed_504_rate_pct": float((fc_cnt or fail_closed_504_count) / max(t_cnt or timeout_count, 1) * 100.0) if (t_cnt or timeout_count) > 0 else 0.0,
        "async_total_quarantined": as_tot or async_total_quarantined,
        "async_malicious_exposed_count": as_mal or async_malicious_exposed_count,
        "hold_fail_open_malicious_exposed_count": h_mal or hold_fail_open_malicious_exposed_count,
        "total_malicious_exposed_count": tot_malicious_exposed,
        "async_malicious_exposure_rate_pct": float((as_mal or async_malicious_exposed_count) / max(as_tot or async_total_quarantined, 1) * 100.0),
        "tier2_held_count": t2_held or tier2_held_count,
        "tier2_not_held_count": t2_not_held or tier2_not_held_count,
        "tier2_service_time_ms": {
            "mean": float(np.mean(t2_st_arr)),
            "p50": float(np.percentile(t2_st_arr, 50)),
            "p95": float(np.percentile(t2_st_arr, 95)),
            "p99": float(np.percentile(t2_st_arr, 99)),
            "min": float(np.min(t2_st_arr)),
            "max": float(np.max(t2_st_arr)),
        },
        "async_exposure_window_ms": {
            "mean": float(np.mean(exp_arr)),
            "p50": float(np.percentile(exp_arr, 50)),
            "p95": float(np.percentile(exp_arr, 95)),
            "max": float(np.max(exp_arr)),
        },
        "server_latencies_ms": {
            "fast_mean": float(mean_f),
            "fast_count": int(tot_fast_cnt or latency_count_fast),
            "esc_mean": float(mean_e),
            "esc_count": int(tot_esc_cnt or latency_count_esc),
            "overall_mean": float(comb_mean),
            "total_count": int(comb_cnt),
        },
    }

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
        print("[Gateway] Verifier ready for low-latency edge screening.")
    return global_verifier


def log_request_sync(method: str, path: str, latency_ms: float, decision: str, escalated: int, mode: str):
    if os.environ.get("GATEWAY_DISABLE_LOGGING", "0") == "1":
        return
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5.0)
        c = conn.cursor()
        c.execute(
            "INSERT INTO access_logs VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
            (method, path, latency_ms, decision, escalated, mode),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


audit_queue: asyncio.Queue = asyncio.Queue()
_audit_worker_task: Optional[asyncio.Task] = None


async def audit_logger_worker():
    while True:
        try:
            batch = []
            item = await audit_queue.get()
            batch.append(item)
            while len(batch) < 200:
                try:
                    item = audit_queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break

            conn = sqlite3.connect(DB_PATH, timeout=15.0)
            c = conn.cursor()
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA busy_timeout=10000;")
            c.executemany(
                """INSERT OR REPLACE INTO request_audit_logs VALUES (
                    ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                batch,
            )
            conn.commit()
            conn.close()
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.02)


def start_audit_worker():
    global _audit_worker_task
    if _audit_worker_task is None or _audit_worker_task.done():
        try:
            loop = asyncio.get_running_loop()
            _audit_worker_task = loop.create_task(audit_logger_worker())
        except RuntimeError:
            pass


def log_request_audit(
    request_id: str,
    ground_truth: int,
    tier1_decision: str,
    path: str,
    tier2_verdict: Optional[str],
    service_time_ms: float,
    status_code: int,
    reached_origin: int,
    malicious_to_origin: int,
    timed_out: int,
    fail_open: int,
    fail_closed_504: int,
    run_tag: str,
):
    if os.environ.get("GATEWAY_DISABLE_AUDIT_LOG", "0") == "1":
        return
    start_audit_worker()
    try:
        audit_queue.put_nowait((
            str(request_id),
            int(ground_truth),
            str(tier1_decision),
            str(path),
            str(tier2_verdict) if tier2_verdict is not None else None,
            float(service_time_ms),
            int(status_code),
            int(reached_origin),
            int(malicious_to_origin),
            int(timed_out),
            int(fail_open),
            int(fail_closed_504),
            str(run_tag),
        ))
    except Exception:
        pass


def query_audit_log_metrics(run_tag: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path, timeout=5.0)
    c = conn.cursor()
    c.execute("""
        SELECT 
            COUNT(*),
            COALESCE(SUM(timed_out), 0),
            COALESCE(SUM(fail_open), 0),
            COALESCE(SUM(fail_closed_504), 0),
            COALESCE(SUM(malicious_to_origin), 0),
            COALESCE(AVG(service_time_ms), 0.0),
            COALESCE(SUM(ground_truth), 0)
        FROM request_audit_logs
        WHERE run_tag = ?
    """, (run_tag,))
    row = c.fetchone()
    conn.close()
    tot = int(row[0]) if row and row[0] is not None else 0
    timeouts = int(row[1]) if row and row[1] is not None else 0
    fwd_unv = int(row[2]) if row and row[2] is not None else 0
    c_504 = int(row[3]) if row and row[3] is not None else 0
    mal_origin = int(row[4]) if row and row[4] is not None else 0
    mean_st = float(row[5]) if row and row[5] is not None else 0.0
    tot_attacks = int(row[6]) if row and row[6] is not None else 0

    return {
        "run_tag": run_tag,
        "total_requests": tot,
        "total_attacks": tot_attacks,
        "timeout_count": timeouts,
        "fail_open_count": fwd_unv,
        "fail_closed_504_count": c_504,
        "malicious_to_origin_count": mal_origin,
        "timeout_share_pct": (timeouts / max(tot, 1)) * 100.0,
        "fail_open_share_pct": (fwd_unv / max(tot, 1)) * 100.0,
        "fail_closed_504_share_pct": (c_504 / max(tot, 1)) * 100.0,
        "malicious_to_origin_share_pct": (mal_origin / max(tot_attacks, 1)) * 100.0,
        "mean_service_time_ms": mean_st,
    }



async def screen_request(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Core Ingress Proxy Screening Logic (Algorithm 1).
    """
    global total_requests, escalated_requests, timeout_count, fail_open_count, fail_closed_504_count, async_total_quarantined
    global latency_sum_fast, latency_count_fast, latency_sum_esc, latency_count_esc
    total_requests += 1
    start_time = time.perf_counter()

    # Synchronize state across workers
    st = load_gateway_state()
    current_mode = st.get("mode", INSPECTION_MODE)
    current_behavior = st.get("behavior", ESCALATION_BEHAVIOR)
    current_timeout_ms = float(st.get("timeout_ms", ESCALATION_TIMEOUT_MS))
    current_timeout_action = st.get("timeout_action", TIMEOUT_ACTION)

    # Extract request properties
    method = request.method
    path = request.url.path
    headers = dict(request.headers)
    req_id = f"{time.time()}_{id(request)}"

    # Determine ground truth label from client header or path heuristic
    gt_header = headers.get("x-ground-truth")
    if gt_header is not None:
        try:
            ground_truth = int(gt_header)
        except ValueError:
            ground_truth = 1 if str(gt_header).lower() in ("1", "attack", "malicious", "true") else 0
    else:
        if "editar.jsp" in path or "registro" in path or "select" in path.lower() or "union" in path.lower():
            ground_truth = 1
        elif any(fp in path for fp in ("/tienda1/publico", "/tienda1/global", "/healthz")):
            ground_truth = 0
        else:
            ground_truth = 1 if method == "POST" else 0

    run_tag = headers.get("x-run-tag", "default")

    # Mode 1: No Inspection Passthrough Baseline
    if current_mode == "no_inspection":
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latency_sum_fast += latency_ms
        latency_count_fast += 1
        mal_to_orig = 1 if ground_truth == 1 else 0
        background_tasks.add_task(
            log_request_audit,
            req_id, ground_truth, "no_inspection", path, None,
            latency_ms, 200, 1, mal_to_orig, 0, 0, 0, run_tag
        )
        return JSONResponse({"status": "Forwarded upstream (no inspection)", "latency_ms": latency_ms})

    body = ""
    try:
        raw_body = await request.body()
        body = raw_body.decode("utf-8", errors="ignore")
    except Exception:
        pass

    url_str = f"http://localhost:8080{path}" if ("HTTP/" in path or path.endswith("HTTP/1.1")) else f"http://localhost:8080{path} HTTP/1.1"
    ua = headers.get("user-agent", "")
    if not ua or "wrk" in ua.lower() or "python" in ua.lower():
        ua = "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)"

    row_data = {
        "Method": method,
        "URL": url_str,
        "content": body,
        "host": headers.get("host", "localhost:8080"),
        "User-Agent": ua,
        "Pragma": headers.get("pragma", "no-cache"),
        "Cache-Control": headers.get("cache-control", "no-cache"),
        "Accept": headers.get("accept", "text/html,application/xhtml+xml,text/xml;q=0.9,*/*;q=0.8"),
        "Accept-encoding": headers.get("accept-encoding", "gzip, deflate"),
        "Accept-charset": headers.get("accept-charset", "utf-8,*;q=0.5"),
        "language": headers.get("language", headers.get("accept-language", "en")),
        "cookie": headers.get("cookie", "JSESSIONID=0BA169D320526F2E099D73D78771389D"),
        "content-type": headers.get("content-type", "application/x-www-form-urlencoded" if method == "POST" else ""),
        "connection": headers.get("connection", "keep-alive"),
    }
    req_row = pd.Series(row_data)

    verifier = get_verifier()

    # Mode 2: Pure Tier 2 (Every request undergoes deep inspection)
    if current_mode == "pure_tier2":
        raw_text = extract_raw_http_text(req_row)
        X2 = verifier.vectorizer.transform([raw_text]).toarray()
        pred = int(verifier.tier2_model.predict(X2)[0])
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latency_sum_esc += latency_ms
        latency_count_esc += 1
        decision = "BLOCKED" if pred == 1 else "FORWARDED"
        background_tasks.add_task(log_request_sync, method, path, latency_ms, decision, 1, "pure_tier2")

        status_code = 403 if pred == 1 else 200
        reached_orig = 1 if pred == 0 else 0
        mal_to_orig = 1 if (pred == 0 and ground_truth == 1) else 0
        t2_v = "attack" if pred == 1 else "benign"
        background_tasks.add_task(
            log_request_audit,
            req_id, ground_truth, "pure_tier2", path, t2_v,
            latency_ms, status_code, reached_orig, mal_to_orig, 0, 0, 0, run_tag
        )

        if pred == 1:
            return JSONResponse(status_code=403, content={"detail": "Blocked by Pure Tier 2 Deep Verification", "latency_ms": latency_ms})
        return JSONResponse({"status": "Forwarded upstream (pure tier 2 verified)", "latency_ms": latency_ms})

    # Mode 3: Two-Tier (Algorithm 1)
    # 1. Fast Feature Extraction
    t1_feats = np.array([extract_track_a_fast_features(req_row)], dtype=np.float32)
    p_attack = float(verifier.predict_proba_fast(t1_feats))

    # Escalation check
    low_band, high_band = verifier.t1_band
    is_uncertain = (p_attack >= low_band) and (p_attack <= high_band)

    if not is_uncertain:
        # Decided via low-latency Tier 1 edge screening
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latency_sum_fast += latency_ms
        latency_count_fast += 1
        if p_attack > high_band:
            background_tasks.add_task(log_request_sync, method, path, latency_ms, "BLOCKED", 0, "two_tier")
            background_tasks.add_task(
                log_request_audit,
                req_id, ground_truth, "direct_block", path, None,
                latency_ms, 403, 0, 0, 0, 0, 0, run_tag
            )
            return JSONResponse(status_code=403, content={"detail": "Blocked by Tier 1 Fast Edge Classifier", "latency_ms": latency_ms})

        background_tasks.add_task(log_request_sync, method, path, latency_ms, "FORWARDED", 0, "two_tier")
        mal_to_orig = 1 if ground_truth == 1 else 0
        background_tasks.add_task(
            log_request_audit,
            req_id, ground_truth, "fast_path", path, None,
            latency_ms, 200, 1, mal_to_orig, 0, 0, 0, run_tag
        )
        return JSONResponse({"status": "Forwarded upstream (Tier 1 confident benign)", "latency_ms": latency_ms})

    # Escalated request
    escalated_requests += 1
    event = asyncio.Event()
    inflight_events[req_id] = event

    # Enqueue to background Tier 2 worker
    await escalation_queue.put({
        "req_id": req_id,
        "row": req_row,
        "t_inbound": start_time,
        "behavior": current_behavior,
    })

    # Handle escalation according to configurable policy
    if current_behavior == "async_quarantine":
        async_total_quarantined += 1
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latency_sum_fast += latency_ms
        latency_count_fast += 1
        background_tasks.add_task(log_request_sync, method, path, latency_ms, "QUARANTINED_ASYNC", 1, "two_tier")
        mal_to_orig = 1 if ground_truth == 1 else 0
        background_tasks.add_task(
            log_request_audit,
            req_id, ground_truth, "escalate", path, "quarantined_async",
            latency_ms, 202, 1, mal_to_orig, 0, 0, 0, run_tag
        )
        return JSONResponse(
            status_code=202,
            content={"status": "Quarantined: forwarded to Tier 2 for async verification", "latency_ms": latency_ms},
        )

    # Otherwise: 'hold_with_timeout'
    try:
        timeout_sec = current_timeout_ms / 1000.0
        await asyncio.wait_for(event.wait(), timeout=timeout_sec)
        t2_verdict = tier2_verdicts.pop(req_id, 0)
        inflight_events.pop(req_id, None)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latency_sum_esc += latency_ms
        latency_count_esc += 1
        status_code = 403 if t2_verdict == 1 else 200
        reached_orig = 1 if t2_verdict == 0 else 0
        mal_to_orig = 1 if (t2_verdict == 0 and ground_truth == 1) else 0
        t2_v = "attack" if t2_verdict == 1 else "benign"

        background_tasks.add_task(
            log_request_audit,
            req_id, ground_truth, "escalate", path, t2_v,
            latency_ms, status_code, reached_orig, mal_to_orig, 0, 0, 0, run_tag
        )

        if t2_verdict == 1:
            background_tasks.add_task(log_request_sync, method, path, latency_ms, "BLOCKED", 1, "two_tier")
            return JSONResponse(status_code=403, content={"detail": "Blocked by Tier 2 Deep Verification", "latency_ms": latency_ms})

        background_tasks.add_task(log_request_sync, method, path, latency_ms, "FORWARDED", 1, "two_tier")
        return JSONResponse({"status": "Forwarded upstream (Tier 2 verified benign)", "latency_ms": latency_ms})

    except asyncio.TimeoutError:
        inflight_events.pop(req_id, None)
        timeout_count += 1
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latency_sum_esc += latency_ms
        latency_count_esc += 1
        if current_timeout_action == "fail_open":
            fail_open_count += 1
            timed_out_fail_open_reqs.add(req_id)
            background_tasks.add_task(log_request_sync, method, path, latency_ms, "TIMEOUT_FAIL_OPEN", 1, "two_tier")
            mal_to_orig = 1 if ground_truth == 1 else 0
            background_tasks.add_task(
                log_request_audit,
                req_id, ground_truth, "escalate", path, "timeout_fail_open",
                latency_ms, 200, 1, mal_to_orig, 1, 1, 0, run_tag
            )
            return JSONResponse(status_code=200, content={"status": "Forwarded upstream (Tier 2 timeout fail-open)", "latency_ms": latency_ms})
        else:
            fail_closed_504_count += 1
            background_tasks.add_task(log_request_sync, method, path, latency_ms, "TIMEOUT_FAIL_CLOSED", 1, "two_tier")
            background_tasks.add_task(
                log_request_audit,
                req_id, ground_truth, "escalate", path, "timeout_fail_closed",
                latency_ms, 504, 0, 0, 1, 0, 1, run_tag
            )
            return JSONResponse(status_code=504, content={"detail": "Tier 2 timeout: Blocked (fail-closed)", "latency_ms": latency_ms})


