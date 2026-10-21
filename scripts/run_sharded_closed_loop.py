#!/usr/bin/env python3
"""
scripts/run_sharded_closed_loop.py
───────────────────────────────────
Executes the closed-loop load benchmark once (Single Run) with the load client sharded
into >= 4 processes (4 processes, each with concurrency C = 5; total C = 20).
Monitors per-process CPU utilization with psutil and verifies that each process CPU is strictly < 60%.
Measures throughput, arithmetic mean latency from per-request timestamps, percentiles,
and verifies Little's Law consistency (|20 - N_est| / 20 < 15%) across all four operational conditions:
  1. no_inspection (Passthrough)
  2. benign_fast_path (Tier 1 Edge Fast-Path)
  3. escalated (Saturated Tier 2 Deep Verification)
  4. two_tier_blended (10% Injected Escalation Share)
"""

import sys
import os
import time
import json
import random
import asyncio
import multiprocessing as mp
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import aiohttp
import numpy as np
import psutil

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from load_test.load_test_suite import GatewayServiceManager


def _worker_process_entry(
    proc_idx: int,
    concurrency_per_proc: int,
    start_barrier: mp.Barrier,
    duration_sec: float,
    cond_name: str,
    url_base: str,
    escalation_rate: float,
    seed: int,
    run_tag: str,
):
    """
    Child process entrypoint: runs an asyncio event loop with concurrency_per_proc
    strict 1-in-flight HTTP connection workers.
    """
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

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

    async def _async_runner():
        all_latencies = []
        timeout = aiohttp.ClientTimeout(total=30.0)
        connector = aiohttp.TCPConnector(
            limit=concurrency_per_proc * 2,
            force_close=False,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
        )

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Pre-warm connection pool
            for _ in range(concurrency_per_proc):
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
            get_urls = [f"{url_base}{p}" for p in fast_paths]
            post_url = f"{url_base}/tienda1/miembros/editar.jsp"

            # Wait for all processes to be ready and start simultaneously
            start_barrier.wait()
            t_start = time.perf_counter()
            t_end = t_start + duration_sec

            async def _connection_task(conn_id: int):
                worker_rng = random.Random(seed * 1000 + proc_idx * 100 + conn_id)
                latencies = []
                req_idx = conn_id
                n_urls = len(get_urls)

                while time.perf_counter() < t_end:
                    is_esc = False
                    if cond_name == "escalated":
                        is_esc = True
                    elif cond_name == "two_tier_blended":
                        is_esc = (worker_rng.random() < escalation_rate)

                    t0 = time.perf_counter()
                    try:
                        if is_esc:
                            async with session.post(
                                post_url,
                                data=esc_body,
                                headers=post_headers,
                            ) as resp:
                                await resp.read()
                        else:
                            url = get_urls[req_idx % n_urls]
                            req_idx += 1
                            async with session.get(url, headers=get_headers) as resp:
                                await resp.read()
                        t1 = time.perf_counter()
                        latencies.append((t1 - t0) * 1000.0)
                    except Exception:
                        pass
                return latencies

            tasks = [_connection_task(c_id) for c_id in range(concurrency_per_proc)]
            results = await asyncio.gather(*tasks)
            for res in results:
                all_latencies.extend(res)

            t_finish = time.perf_counter()

        meta_file = f"/tmp/sharded_meta_{run_tag}_{proc_idx}.json"
        lat_file = f"/tmp/sharded_lat_{run_tag}_{proc_idx}.npy"
        np.save(lat_file, np.array(all_latencies, dtype=np.float32))
        with open(meta_file, "w") as f:
            json.dump({
                "proc_idx": proc_idx,
                "lat_file": lat_file,
                "req_count": len(all_latencies),
                "t_start": t_start,
                "t_finish": t_finish,
            }, f)

    asyncio.run(_async_runner())


def run_sharded_closed_loop_condition(
    cond_name: str,
    gw_mode: str,
    url_base: str = "http://127.0.0.1:8080",
    num_processes: int = 4,
    concurrency_per_proc: int = 5,
    duration_sec: float = 4.0,
    escalation_rate: float = 0.10,
    seed: int = 42,
    run_tag: str = "sharded_cl",
) -> Dict[str, Any]:
    """
    Executes a single closed-loop benchmark run sharded across num_processes independent processes.
    Monitors per-process CPU utilization and records all individual request timestamps.
    """
    total_concurrency = num_processes * concurrency_per_proc
    barrier = mp.Barrier(num_processes + 1)
    processes: List[mp.Process] = []

    # Clean up temp files
    for p_idx in range(num_processes):
        for pattern in [f"/tmp/sharded_meta_{run_tag}_{p_idx}.json", f"/tmp/sharded_lat_{run_tag}_{p_idx}.npy"]:
            if os.path.exists(pattern):
                try:
                    os.remove(pattern)
                except Exception:
                    pass

    # Spawn sharded client processes
    for p_idx in range(num_processes):
        p = mp.Process(
            target=_worker_process_entry,
            args=(
                p_idx,
                concurrency_per_proc,
                barrier,
                duration_sec,
                cond_name,
                url_base,
                escalation_rate,
                seed + p_idx * 17,
                run_tag,
            ),
        )
        processes.append(p)
        p.start()

    ps_procs = [psutil.Process(p.pid) for p in processes]
    # Prime CPU sampling
    for ps_p in ps_procs:
        try:
            ps_p.cpu_percent(interval=None)
        except Exception:
            pass

    # Release barrier so all workers start together
    barrier.wait()

    proc_cpu_samples: Dict[int, List[float]] = {p.pid: [] for p in ps_procs}
    t_end_monitor = time.time() + duration_sec + 1.5

    while any(p.is_alive() for p in processes) and time.time() < t_end_monitor:
        for ps_p in ps_procs:
            try:
                if ps_p.is_running():
                    proc_cpu_samples[ps_p.pid].append(ps_p.cpu_percent(interval=None))
            except Exception:
                pass
        time.sleep(0.1)

    for p in processes:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

    # Collect results from all processes
    all_latencies = []
    start_times = []
    finish_times = []

    for p_idx in range(num_processes):
        meta_file = f"/tmp/sharded_meta_{run_tag}_{p_idx}.json"
        if os.path.exists(meta_file):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                start_times.append(meta["t_start"])
                finish_times.append(meta["t_finish"])
                lat_f = meta.get("lat_file")
                if lat_f and os.path.exists(lat_f):
                    lats = np.load(lat_f)
                    all_latencies.extend(lats.tolist())
                    os.remove(lat_f)
                os.remove(meta_file)
            except Exception as e:
                print(f"[Error reading meta {p_idx}]: {e}")

    # Compute CPU stats per process
    per_proc_cpu = []
    for p in processes:
        samples = proc_cpu_samples.get(p.pid, [])
        valid = [s for s in samples[1:] if s > 0.0]
        avg_cpu = float(np.mean(valid)) if valid else (float(np.mean(samples)) if samples else 0.0)
        per_proc_cpu.append(round(avg_cpu, 1))

    total_requests = len(all_latencies)
    if total_requests == 0:
        return {
            "condition": cond_name,
            "throughput_rps": 0.0,
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p90_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "latency_p999_ms": 0.0,
            "n_est": 0.0,
            "littles_law_error_pct": 100.0,
            "littles_law_passed": False,
            "per_process_cpu_pct": per_proc_cpu,
            "max_process_cpu_pct": max(per_proc_cpu) if per_proc_cpu else 0.0,
            "client_cpu_under_threshold": False,
        }

    actual_duration = float(np.mean([f - s for s, f in zip(start_times, finish_times)]))
    throughput = float(total_requests / actual_duration)
    lat_arr = np.array(all_latencies, dtype=float)
    mean_lat_ms = float(np.mean(lat_arr))
    p50, p90, p95, p99, p999 = np.percentile(lat_arr, [50, 90, 95, 99, 99.9])

    n_est = throughput * (mean_lat_ms / 1000.0)
    littles_law_err = abs(total_concurrency - n_est) / total_concurrency

    return {
        "condition": cond_name,
        "execution_type": "Single Run",
        "num_processes": num_processes,
        "concurrency_per_process": concurrency_per_proc,
        "total_concurrency": total_concurrency,
        "duration_sec": round(actual_duration, 2),
        "total_requests": total_requests,
        "throughput_rps": round(throughput, 1),
        "latency_mean_ms": round(mean_lat_ms, 2),
        "latency_p50_ms": round(float(p50), 2),
        "latency_p90_ms": round(float(p90), 2),
        "latency_p95_ms": round(float(p95), 2),
        "latency_p99_ms": round(float(p99), 2),
        "latency_p999_ms": round(float(p999), 2),
        "n_est": round(n_est, 2),
        "littles_law_error_pct": round(littles_law_err * 100.0, 2),
        "littles_law_passed": bool(littles_law_err < 0.15),
        "per_process_cpu_pct": per_proc_cpu,
        "max_process_cpu_pct": max(per_proc_cpu),
        "client_cpu_under_threshold": bool(max(per_proc_cpu) < 60.0),
    }


def main():
    print("=" * 80)
    print("SHARDED LOAD CLIENT CLOSED-LOOP BENCHMARK (>=4 PROCESSES, TOTAL C = 20)")
    print("Requirement: 4 client processes, C=5 each, per-process CPU < 60%")
    print("Execution Type: Single Run across all 4 operational conditions")
    print("=" * 80)

    subprocess.run("lsof -ti :8000 -ti :8080 | xargs kill -9 2>/dev/null || true", shell=True)
    time.sleep(1.0)

    gw = GatewayServiceManager(workers=4, disable_onnx=False)
    gw.start()

    conditions = [
        ("no_inspection", "no_inspection", 0.0),
        ("benign_fast_path", "two_tier", 0.0),
        ("escalated", "two_tier", 1.0),
        ("two_tier_blended", "two_tier", 0.10),
    ]

    all_results = {}

    try:
        gw.set_escalation_config(behavior="hold_with_timeout", timeout_action="fail_open", timeout_ms=60.0)

        for cond_name, gw_mode, esc_rate in conditions:
            print(f"\n--- Testing Condition: {cond_name} (Mode: {gw_mode}, Injected ER: {esc_rate*100:.0f}%) ---")
            gw.set_mode(gw_mode)
            gw.reset_state()
            time.sleep(0.5)

            # Warmup run (1.5s)
            run_sharded_closed_loop_condition(
                cond_name=cond_name,
                gw_mode=gw_mode,
                num_processes=4,
                concurrency_per_proc=5,
                duration_sec=1.5,
                escalation_rate=esc_rate,
                seed=999,
                run_tag=f"warmup_{cond_name}",
            )
            time.sleep(0.3)

            # Measurement run (4.0s)
            res = run_sharded_closed_loop_condition(
                cond_name=cond_name,
                gw_mode=gw_mode,
                num_processes=4,
                concurrency_per_proc=5,
                duration_sec=4.0,
                escalation_rate=esc_rate,
                seed=42,
                run_tag=f"sharded_single_{cond_name}",
            )
            all_results[cond_name] = res

            print(f"Results for {cond_name}:")
            print(f"  Throughput: {res['throughput_rps']:.1f} req/s")
            print(f"  Mean Latency: {res['latency_mean_ms']:.2f} ms | p50: {res['latency_p50_ms']:.2f} ms | p95: {res['latency_p95_ms']:.2f} ms")
            print(f"  Little's Law: N_est = {res['n_est']:.2f} (Error: {res['littles_law_error_pct']:.2f}%, Passed: {res['littles_law_passed']})")
            print(f"  Per-Process CPU: {res['per_process_cpu_pct']} (Max: {res['max_process_cpu_pct']:.1f}%, <60%: {res['client_cpu_under_threshold']})")

            assert res["littles_law_passed"], f"Little's Law failed for {cond_name}: error {res['littles_law_error_pct']}% >= 15%"
            assert res["client_cpu_under_threshold"], f"Client CPU failed for {cond_name}: max {res['max_process_cpu_pct']}% >= 60%"

        # Save results
        out_path = PROJECT_ROOT / "results/latency/sharded_closed_loop_run.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved sharded closed-loop single-run results to {out_path}")

    finally:
        gw.stop()
        subprocess.run("lsof -ti :8000 -ti :8080 | xargs kill -9 2>/dev/null || true", shell=True)
        print("Gateway stopped.")


if __name__ == "__main__":
    main()
