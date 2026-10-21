#!/usr/bin/env python3
"""
Test script to verify sharded closed-loop client CPU and Little's Law compliance.
Tests both 4 processes (C=5) and 5 processes (C=4).
"""
import os
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from load_test.load_test_suite import GatewayServiceManager


def _worker_entry(
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

    async def _runner():
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

            # Wait at barrier so all processes start at the exact same moment
            start_barrier.wait()
            t_start = time.perf_counter()
            t_end = t_start + duration_sec

            async def _conn_loop(conn_id: int):
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

            tasks = [_conn_loop(c_id) for c_id in range(concurrency_per_proc)]
            results = await asyncio.gather(*tasks)
            for res in results:
                all_latencies.extend(res)

            t_finish = time.perf_counter()

        meta_file = f"/tmp/test_sharded_meta_{run_tag}_{proc_idx}.json"
        lat_file = f"/tmp/test_sharded_lat_{run_tag}_{proc_idx}.npy"
        np.save(lat_file, np.array(all_latencies, dtype=np.float32))
        with open(meta_file, "w") as f:
            json.dump({
                "proc_idx": proc_idx,
                "lat_file": lat_file,
                "req_count": len(all_latencies),
                "t_start": t_start,
                "t_finish": t_finish,
            }, f)

    asyncio.run(_runner())


def run_condition_test(num_procs: int, conc_per_proc: int):
    total_concurrency = num_procs * conc_per_proc
    run_tag = f"diag_{num_procs}p"
    barrier = mp.Barrier(num_procs + 1)
    duration_sec = 4.0

    procs = []
    for p_idx in range(num_procs):
        p = mp.Process(
            target=_worker_entry,
            args=(
                p_idx,
                conc_per_proc,
                barrier,
                duration_sec,
                "no_inspection",
                "http://127.0.0.1:8080",
                0.0,
                42,
                run_tag,
            ),
        )
        procs.append(p)
        p.start()

    ps_procs = [psutil.Process(p.pid) for p in procs]
    # Initialize CPU measurement
    for p in ps_procs:
        try:
            p.cpu_percent(interval=None)
        except Exception:
            pass

    # Release barrier to start
    barrier.wait()

    proc_cpu_samples: Dict[int, List[float]] = {p.pid: [] for p in ps_procs}
    t_end = time.time() + duration_sec + 1.0

    while any(p.is_alive() for p in procs) and time.time() < t_end:
        for ps_p in ps_procs:
            try:
                if ps_p.is_running():
                    proc_cpu_samples[ps_p.pid].append(ps_p.cpu_percent(interval=None))
            except Exception:
                pass
        time.sleep(0.1)

    for p in procs:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

    # Collect results
    all_lats = []
    t_starts = []
    t_finishes = []
    for p_idx in range(num_procs):
        m_file = f"/tmp/test_sharded_meta_{run_tag}_{p_idx}.json"
        if os.path.exists(m_file):
            with open(m_file) as f:
                meta = json.load(f)
            t_starts.append(meta["t_start"])
            t_finishes.append(meta["t_finish"])
            lat_f = meta["lat_file"]
            lats = np.load(lat_f)
            all_lats.extend(lats.tolist())
            os.remove(lat_f)
            os.remove(m_file)

    lats_arr = np.array(all_lats, dtype=float)
    actual_dur = np.mean([f - s for s, f in zip(t_starts, t_finishes)])
    tp = len(all_lats) / actual_dur
    mean_lat_ms = float(np.mean(lats_arr))
    n_est = tp * (mean_lat_ms / 1000.0)
    err = abs(total_concurrency - n_est) / total_concurrency

    # CPU stats
    cpu_list = []
    for pid, samples in proc_cpu_samples.items():
        valid = [s for s in samples[1:] if s > 0.0]
        cpu_list.append(round(float(np.mean(valid)), 1) if valid else 0.0)

    print(f"\n--- Diagnostic: {num_procs} processes, C={conc_per_proc} (Total C={total_concurrency}) ---")
    print(f"Total Requests: {len(all_lats)} in {actual_dur:.2f}s")
    print(f"Throughput: {tp:.1f} req/s")
    print(f"Mean Latency: {mean_lat_ms:.2f} ms")
    print(f"Little's Law: N_est = {n_est:.2f} (Error: {err*100:.2f}%, Passed: {err < 0.15})")
    print(f"Per-Process CPU: {cpu_list} (Max: {max(cpu_list)}%, <60%: {max(cpu_list) < 60.0})")


def main():
    subprocess.run("lsof -ti :8000,:8080 | xargs kill -9 2>/dev/null || true", shell=True)
    time.sleep(1.0)
    gw = GatewayServiceManager(workers=4)
    gw.start()
    try:
        gw.set_mode("no_inspection")
        time.sleep(0.5)
        print("Testing 4 processes (C=5 each)...")
        run_condition_test(num_procs=4, conc_per_proc=5)

        time.sleep(1.0)
        print("\nTesting 5 processes (C=4 each)...")
        run_condition_test(num_procs=5, conc_per_proc=4)

    finally:
        gw.stop()
        subprocess.run("lsof -ti :8000,:8080 | xargs kill -9 2>/dev/null || true", shell=True)


if __name__ == "__main__":
    main()
