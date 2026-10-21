#!/usr/bin/env python3
"""
scripts/run_overload_benchmark.py
───────────────────────────────────
Executes the overload benchmark at 1.5x measured Tier 2 capacity (1.5 * 516.0 = 774.0 req/s)
using an attack-heavy stream (85% attack / 15% benign).

Evaluates 4 controller configurations:
  1. no_controller_baseline (Timeout controller disabled / 10s budget, no drop)
  2. hold_fail_open (60ms timeout budget, fail-open forwarded unverified)
  3. hold_fail_closed (60ms timeout budget, fail-closed 504 Gateway Timeout)
  4. async_quarantine (202 Accepted quarantine, async Tier 2 inspection)

Derives all metrics directly from SQLite `request_audit_logs` using `query_audit_log_metrics`:
  - Total requests
  - Timeout share of ALL requests (%)
  - Fail-open exposure share of ALL requests (%)
  - Fail-closed 504 share of ALL requests (%)
  - Malicious requests reaching origin (count & share %)
  - Mean service time (ms)
"""

import sys
import os
import time
import json
import random
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import aiohttp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from load_test.load_test_suite import GatewayServiceManager
from src.gateway.middleware import query_audit_log_metrics


TIER2_MEASURED_CAPACITY_RPS = 516.0
OVERLOAD_FACTOR = 1.50
OFFERED_LOAD_RPS = round(TIER2_MEASURED_CAPACITY_RPS * OVERLOAD_FACTOR, 1)  # 774.0 req/s
TEST_DURATION_SEC = 8.0
ATTACK_STREAM_SHARE = 0.85  # Attack-heavy stream


async def run_open_loop_overload_stream(
    target_rps: float,
    duration_sec: float,
    attack_share: float,
    url_base: str,
    run_tag: str,
) -> Dict[str, Any]:
    """
    Generates an open-loop Poisson/constant-rate offered load stream at target_rps.
    Uses an attack-heavy stream composition and tags each request for audit logging.
    """
    fast_paths = [
        "/tienda1/global/creditos.jsp",
        "/tienda1/publico/carrito.jsp",
        "/tienda1/publico/miembros.jsp",
        "/tienda1/publico/productos.jsp",
    ]
    esc_body = (
        "modo=registro&login=glaser&password=elementada&nombre=Davel&apellidos=De+Mendonca+Ors&"
        "email=chua%40comerciosdeaspe.om&dni=41909752A&direccion=De+La+Coma%2C+112+4-B&"
        "ciudad=San+Miguel+de+Abona&cp=04100&provincia=Ourense&ntc=4150902070293588&B1=Registrar"
    )

    timeout = aiohttp.ClientTimeout(total=15.0)
    connector = aiohttp.TCPConnector(limit=500, force_close=False, keepalive_timeout=60)

    total_requests_scheduled = int(target_rps * duration_sec)
    inter_arrival = 1.0 / target_rps

    latencies = []
    status_codes = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Pre-warm
        for _ in range(10):
            try:
                async with session.get(f"{url_base}/healthz") as r:
                    await r.read()
            except Exception:
                pass

        get_headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)",
            "Accept": "text/html,application/xhtml+xml,text/xml;q=0.9,*/*;q=0.8",
            "X-Run-Tag": run_tag,
            "X-Ground-Truth": "0",
        }
        post_headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)",
            "Accept": "text/html,application/xhtml+xml,text/xml;q=0.9,*/*;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Run-Tag": run_tag,
            "X-Ground-Truth": "1",
        }
        post_url = f"{url_base}/tienda1/miembros/editar.jsp"

        rng = random.Random(42)

        async def _send_one(is_attack: bool, req_id: int):
            t0 = time.perf_counter()
            code = 0
            try:
                if is_attack:
                    async with session.post(post_url, data=esc_body, headers=post_headers) as resp:
                        code = resp.status
                        await resp.read()
                else:
                    p = fast_paths[req_id % len(fast_paths)]
                    async with session.get(f"{url_base}{p}", headers=get_headers) as resp:
                        code = resp.status
                        await resp.read()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
                status_codes.append(code)
            except asyncio.TimeoutError:
                status_codes.append(504)
                latencies.append(15000.0)
            except Exception:
                status_codes.append(0)

        # Launch requests at scheduled timestamps
        t_start = time.perf_counter()
        tasks = []

        for req_idx in range(total_requests_scheduled):
            target_time = t_start + req_idx * inter_arrival
            now = time.perf_counter()
            delay = target_time - now
            if delay > 0.0005:
                await asyncio.sleep(delay)

            is_att = (rng.random() < attack_share)
            t = asyncio.create_task(_send_one(is_att, req_idx))
            tasks.append(t)

        await asyncio.gather(*tasks, return_exceptions=True)
        t_finish = time.perf_counter()

    actual_duration = t_finish - t_start
    achieved_rps = len(latencies) / actual_duration if actual_duration > 0 else 0.0

    return {
        "scheduled_requests": total_requests_scheduled,
        "completed_requests": len(latencies),
        "actual_duration_sec": actual_duration,
        "achieved_rps": achieved_rps,
        "client_latencies_ms": latencies,
        "status_codes": status_codes,
    }


def main():
    print("=" * 85)
    print(f"TIER 2 OVERLOAD BENCHMARK (1.5x CAPACITY = {OFFERED_LOAD_RPS} req/s)")
    print(f"Stream Composition: {ATTACK_STREAM_SHARE*100:.0f}% Attack / {(1-ATTACK_STREAM_SHARE)*100:.0f}% Benign")
    print("Measurement Type: Single Run (Audit Log Ground-Truth Derivation)")
    print("=" * 85)

    subprocess.run("lsof -ti :8000 -ti :8080 | xargs kill -9 2>/dev/null || true", shell=True)
    time.sleep(1.0)

    gw = GatewayServiceManager(workers=4, disable_onnx=False)
    gw.start()

    configurations = [
        ("no_controller_baseline", "hold_with_timeout", "fail_open", 10000.0),
        ("hold_fail_open", "hold_with_timeout", "fail_open", 60.0),
        ("hold_fail_closed", "hold_with_timeout", "fail_closed", 60.0),
        ("async_quarantine", "async_quarantine", "fail_open", 60.0),
    ]

    overload_results = {}

    try:
        gw.set_mode("two_tier")

        for cond_name, behavior, timeout_action, timeout_ms in configurations:
            print(f"\n--- Testing Overload Condition: {cond_name} ---")
            print(f"    Behavior: {behavior} | Timeout Action: {timeout_action} | Timeout: {timeout_ms} ms")
            gw.set_escalation_config(behavior=behavior, timeout_action=timeout_action, timeout_ms=timeout_ms)
            gw.reset_state()
            time.sleep(0.5)

            run_tag = f"overload_{cond_name}"

            # Execute open-loop overload stream
            stream_res = asyncio.run(run_open_loop_overload_stream(
                target_rps=OFFERED_LOAD_RPS,
                duration_sec=TEST_DURATION_SEC,
                attack_share=ATTACK_STREAM_SHARE,
                url_base="http://127.0.0.1:8080",
                run_tag=run_tag,
            ))

            # Allow audit log queue to completely flush to disk
            time.sleep(1.5)

            # Query ground-truth audit log metrics directly from SQLite
            audit_metrics = query_audit_log_metrics(run_tag)

            lats = stream_res["client_latencies_ms"]
            p50 = float(np.percentile(lats, 50)) if lats else 0.0
            p95 = float(np.percentile(lats, 95)) if lats else 0.0
            p99 = float(np.percentile(lats, 99)) if lats else 0.0
            mean_lat = float(np.mean(lats)) if lats else 0.0

            row_summary = {
                "condition": cond_name,
                "execution_type": "Single Run",
                "offered_load_rps": OFFERED_LOAD_RPS,
                "achieved_rps": round(stream_res["achieved_rps"], 1),
                "duration_sec": round(stream_res["actual_duration_sec"], 2),
                "stream_composition": f"{ATTACK_STREAM_SHARE*100:.0f}% Attack / {(1-ATTACK_STREAM_SHARE)*100:.0f}% Benign",
                "total_requests": audit_metrics["total_requests"],
                "total_attacks": audit_metrics["total_attacks"],
                "timeout_count": audit_metrics["timeout_count"],
                "timeout_share_pct": round(audit_metrics["timeout_share_pct"], 2),
                "fail_open_count": audit_metrics["fail_open_count"],
                "fail_open_share_pct": round(audit_metrics["fail_open_share_pct"], 2),
                "fail_closed_504_count": audit_metrics["fail_closed_504_count"],
                "fail_closed_504_share_pct": round(audit_metrics["fail_closed_504_share_pct"], 2),
                "malicious_to_origin_count": audit_metrics["malicious_to_origin_count"],
                "malicious_to_origin_share_pct": round(audit_metrics["malicious_to_origin_share_pct"], 2),
                "mean_service_time_ms": round(audit_metrics["mean_service_time_ms"], 2),
                "client_latency_p50_ms": round(p50, 2),
                "client_latency_p95_ms": round(p95, 2),
                "client_latency_p99_ms": round(p99, 2),
                "client_latency_mean_ms": round(mean_lat, 2),
            }

            overload_results[cond_name] = row_summary

            print(f"  Total Requests Logged: {row_summary['total_requests']} (Attacks: {row_summary['total_attacks']})")
            print(f"  Timeout Share of ALL Requests: {row_summary['timeout_share_pct']}% ({row_summary['timeout_count']} reqs)")
            print(f"  Fail-Open Exposure Share: {row_summary['fail_open_share_pct']}% ({row_summary['fail_open_count']} reqs)")
            print(f"  Fail-Closed 504 Share: {row_summary['fail_closed_504_share_pct']}% ({row_summary['fail_closed_504_count']} reqs)")
            print(f"  Malicious-to-Origin: {row_summary['malicious_to_origin_count']} reqs ({row_summary['malicious_to_origin_share_pct']}%)")
            print(f"  Client Latency Mean: {row_summary['client_latency_mean_ms']} ms | p95: {row_summary['client_latency_p95_ms']} ms")

        out_path = PROJECT_ROOT / "results/latency/overload_run.json"
        with open(out_path, "w") as f:
            json.dump(overload_results, f, indent=2)
        print(f"\nSaved overload benchmark results to {out_path}")

    finally:
        gw.stop()
        subprocess.run("lsof -ti :8000 -ti :8080 | xargs kill -9 2>/dev/null || true", shell=True)
        print("Gateway stopped.")


if __name__ == "__main__":
    main()
