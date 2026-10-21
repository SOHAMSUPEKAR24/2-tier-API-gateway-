#!/usr/bin/env python3
"""
Runs the worker and backend ablation using the EXACT SAME aiohttp closed-loop harness (C = 20)
as the closed-loop table, monitoring client process count and CPU usage.
"""
import asyncio
import os
import time
import json
import psutil
from pathlib import Path
from load_test.load_test_suite import GatewayServiceManager, _execute_aiohttp_closed_loop_run

def run_reconciled_ablation():
    configs = [
        ("sklearn_1worker", 1, True, "Raw Scikit-Learn (1 Worker)"),
        ("onnx_1worker", 1, False, "Compiled ONNX Forest (1 Worker)"),
        ("onnx_4workers", 4, False, "Compiled ONNX Forest (4 Workers)"),
    ]

    results = []
    client_proc = psutil.Process(os.getpid())

    for key, workers, disable_onnx, label in configs:
        print(f"\n--- Testing {label} (workers={workers}, disable_onnx={disable_onnx}) ---")
        gw = GatewayServiceManager(workers=workers, disable_onnx=disable_onnx)
        gw.start()
        try:
            gw.set_mode("two_tier")
            gw.reset_state()
            time.sleep(0.5)

            # Warmup
            asyncio.run(_execute_aiohttp_closed_loop_run(
                cond_name="benign_fast_path",
                gw_mode="two_tier",
                url_base="http://127.0.0.1:8080",
                concurrency=20,
                duration_sec=1,
                escalation_rate=0.0,
                seed=42,
            ))
            time.sleep(0.2)

            # Measurement run with client CPU profiling
            cpu_before = client_proc.cpu_times()
            t_start = time.perf_counter()
            m = asyncio.run(_execute_aiohttp_closed_loop_run(
                cond_name="benign_fast_path",
                gw_mode="two_tier",
                url_base="http://127.0.0.1:8080",
                concurrency=20,
                duration_sec=3,
                escalation_rate=0.0,
                seed=42,
            ))
            t_elapsed = time.perf_counter() - t_start
            cpu_after = client_proc.cpu_times()

            user_cpu = cpu_after.user - cpu_before.user
            sys_cpu = cpu_after.system - cpu_before.system
            total_client_cpu_sec = user_cpu + sys_cpu
            client_cpu_utilization_pct = (total_client_cpu_sec / t_elapsed) * 100.0

            m["config_key"] = key
            m["label"] = label
            m["workers"] = workers
            m["backend"] = "sklearn" if disable_onnx else "onnx"
            m["concurrency"] = 20
            m["nginx_in_path"] = True
            m["client_process_count"] = 1
            m["client_threads"] = client_proc.num_threads()
            m["client_cpu_pct"] = round(client_cpu_utilization_pct, 1)

            results.append(m)
            print(f"  Result: {m['throughput_rps']:.1f} rps | p50: {m['latency_p50_ms']:.2f}ms | p95: {m['latency_p95_ms']:.2f}ms | Mean: {m['latency_mean_ms']:.2f}ms | Client CPU: {m['client_cpu_pct']}%")
        finally:
            gw.stop()
            time.sleep(1)

    print("\n" + "=" * 78)
    print("RECONCILED ABLATION RESULTS (SAME aiohttp CLIENT, C=20, NGINX IN PATH):")
    for r in results:
        print(f"  {r['label']:32s} | Workers={r['workers']} | TP: {r['throughput_rps']:7.1f} req/s | p50: {r['latency_p50_ms']:5.2f}ms | p95: {r['latency_p95_ms']:5.2f}ms | Client CPU: {r['client_cpu_pct']}%")
    print("=" * 78)

if __name__ == "__main__":
    run_reconciled_ablation()
