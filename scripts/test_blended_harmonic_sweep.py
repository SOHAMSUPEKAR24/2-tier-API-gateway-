#!/usr/bin/env python3
"""
Executes the open-loop blended stream load sweep across rates [250, 500, 750, 1000, 1200, 1400, 1600] req/s
under empirical escalation share (37.7%), collecting p99.9 and timeout counts,
and reconciling empirical saturation with the harmonic queuing prediction (~1,200 req/s).
"""
import subprocess
import time
import json
import random
from pathlib import Path
import numpy as np
from load_test.load_test_suite import GatewayServiceManager, mean_ci95, detect_saturation_knee

def run_blended_sweep():
    rates = [250, 500, 750, 1000, 1200, 1400, 1600]
    repetitions = 5
    duration_sec = 3
    empirical_er = 0.377

    k6_script_path = Path("load_test/run/open_loop_blended.js")
    k6_script = f"""
import http from 'k6/http';
export const options = {{
  scenarios: {{
    constant_load: {{
      executor: 'constant-arrival-rate',
      rate: parseInt(__ENV.TARGET_RATE || '500'),
      timeUnit: '1s',
      duration: `${{__ENV.DURATION || '{duration_sec}'}}s`,
      preAllocatedVUs: 50,
      maxVUs: 400,
    }},
  }},
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)', 'p(99.9)'],
}};

const esc_body = "modo=registro&login=glaser&password=elementada&nombre=Davel&apellidos=De+Mendonca+Ors&email=chua%40comerciosdeaspe.om&dni=41909752A&direccion=De+La+Coma%2C+112+4-B&ciudad=San+Miguel+de+Abona&cp=04100&provincia=Ourense&ntc=4150902070293588&B1=Registrar";

export default function () {{
  const is_escalated = Math.random() < {empirical_er};
  const headers = {{
    'User-Agent': 'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)',
    'Accept': 'text/html,application/xhtml+xml,text/xml;q=0.9,*/*;q=0.8',
  }};

  if (is_escalated) {{
    headers['Content-Type'] = 'application/x-www-form-urlencoded';
    http.post('http://127.0.0.1:8080/tienda1/miembros/editar.jsp', esc_body, {{ headers: headers }});
  }} else {{
    http.get('http://127.0.0.1:8080/tienda1/global/creditos.jsp', {{ headers: headers }});
  }}
}}
"""
    with open(k6_script_path, "w") as f:
        f.write(k6_script)

    gw = GatewayServiceManager(workers=4, disable_onnx=False)
    gw.start()
    try:
        gw.set_mode("two_tier")
        gw.set_escalation_config(behavior="hold_with_timeout", timeout_action="fail_open", timeout_ms=60.0)

        schedule = [(r, rep) for r in rates for rep in range(repetitions)]
        rng = random.Random(137)
        rng.shuffle(schedule)
        print(f"Total scheduled blended runs: {len(schedule)} runs (Randomized order)")

        raw_blended = {r: [] for r in rates}

        for idx, (rate, rep) in enumerate(schedule, 1):
            gw.reset_state()
            time.sleep(0.2)

            out_json = Path(f"load_test/run/k6_blend_harmonic_{rate}_{rep}.json")
            cmd = [
                "k6", "run", "--summary-export", str(out_json),
                "-e", f"TARGET_RATE={rate}",
                "-e", f"DURATION={duration_sec}",
                str(k6_script_path)
            ]
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open(out_json) as f:
                    d = json.load(f)
                dur = d["metrics"]["http_req_duration"]
                rate_achieved = d["metrics"]["http_reqs"]["rate"]
                stats = gw.get_stats()
                m = {
                    "achieved_rate": float(rate_achieved),
                    "p50_ms": float(dur.get("med", 0.0)),
                    "p90_ms": float(dur.get("p(90)", 0.0)),
                    "p95_ms": float(dur.get("p(95)", 0.0)),
                    "p99_ms": float(dur.get("p(99)", 0.0)),
                    "p999_ms": float(dur.get("p(99.9)", dur.get("p(99)", 0.0))),
                    "mean_ms": float(dur.get("avg", 0.0)),
                    "timeout_count": stats.get("timeout_count", 0),
                }
                raw_blended[rate].append(m)
                print(f"  [Run {idx:2d}/{len(schedule)}] @ {rate:4d} rps (rep {rep+1}) -> achieved={m['achieved_rate']:6.1f} | p50={m['p50_ms']:5.2f}ms | p95={m['p95_ms']:5.2f}ms | p99.9={m['p999_ms']:5.2f}ms | timeouts={m['timeout_count']}")
            except Exception as e:
                print(f"  Warning: Blended run failed @ {rate} rps: {e}")

        agg_blended = {}
        for r in rates:
            runs = raw_blended[r]
            if not runs:
                continue
            item = {"offered_rate": r, "runs_count": len(runs)}
            for k in ["achieved_rate", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "p999_ms", "mean_ms", "timeout_count"]:
                vals = [x[k] for x in runs]
                mean_v, ci_v = mean_ci95(vals)
                item[k] = {"mean": float(mean_v), "ci95": float(ci_v)}
            agg_blended[r] = item

        blended_rates = [r for r in rates if r in agg_blended]
        blended_p95s = [agg_blended[r]["p95_ms"]["mean"] for r in blended_rates]
        blended_achieved = [agg_blended[r]["achieved_rate"]["mean"] for r in blended_rates]
        knee_rate, knee_label, knee_reason = detect_saturation_knee(
            blended_rates, blended_p95s, threshold_factor=2.0, achieved_rates=blended_achieved, achieved_ratio_threshold=0.98
        )

        print("\n" + "=" * 78)
        print("BLENDED STREAM OPEN-LOOP LOAD SWEEP RESULTS:")
        print(f"{'Offered':<9} | {'Achieved (rps)':<17} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99.9 (ms)':<10} | {'Mean (ms)':<10} | {'Timeouts':<12}")
        print("-" * 88)
        for r in blended_rates:
            st = agg_blended[r]
            print(f"{r:4d} rps  | {st['achieved_rate']['mean']:6.1f} ± {st['achieved_rate']['ci95']:4.1f} | {st['p50_ms']['mean']:5.2f} ± {st['p50_ms']['ci95']:4.2f} | {st['p95_ms']['mean']:5.2f} ± {st['p95_ms']['ci95']:4.2f} | {st['p999_ms']['mean']:5.2f} ± {st['p999_ms']['ci95']:4.2f} | {st['mean_ms']['mean']:5.2f} ± {st['mean_ms']['ci95']:4.2f} | {st['timeout_count']['mean']:4.1f} ± {st['timeout_count']['ci95']:3.1f}")
        print("=" * 78)
        print(f"KNEE DETECTOR RAW OUTPUT: {knee_label} ({knee_reason})")
        min_p95 = min(blended_p95s)
        print(f"KNEE DETECTOR BASELINE: {min_p95:.2f} ms")

        # Harmonic reconciliation check
        X_fast = 6040.8
        X_t2 = 516.0
        p_esc = 0.377
        X_harmonic = 1.0 / (((1.0 - p_esc) / X_fast) + (p_esc / X_t2))
        print(f"\nHarmonic Queuing Model Prediction:")
        print(f"  1 / X_blend = (1 - p)/X_fast + p/X_t2 = 0.623/{X_fast:.1f} + 0.377/{X_t2:.1f}")
        print(f"  X_harmonic = {X_harmonic:.1f} req/s (~1,200 req/s)")
        print(f"  Empirical saturation knee observed: {knee_label}")
        print("=" * 78)

        # Save results to disk
        with open("load_test/run/blended_harmonic_sweep.json", "w") as f:
            json.dump({
                "rates": rates,
                "aggregated_blended": agg_blended,
                "saturation_knee_rate": knee_rate,
                "saturation_knee_label": knee_label,
                "saturation_knee_reason": knee_reason,
                "baseline_p95_ms": min_p95,
                "harmonic_prediction_rps": X_harmonic,
            }, f, indent=2)

    finally:
        gw.stop()

if __name__ == "__main__":
    run_blended_sweep()
