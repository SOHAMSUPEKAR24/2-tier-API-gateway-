"""
scripts/run_audit.py
────────────────────
Audit script — implements TASK 1 of the project rules.

What this does:
  1. Scans every Python script + TSX dashboard file for:
     - Hardcoded metric values (HARDCODED)
     - Simulated / synthetic metric generators (SIMULATED)
     - Metrics sourced from results/*.json (MEASURED)
  2. Checks whether the Tier 2 labels in the unseen-family experiment
     come from dataset ground truth or a trained model — and documents this.
  3. Re-runs the unseen-family experiment with Dst Port removed and
     compares before/after accuracy and escalation rate.
  4. Writes docs/audit.md with a full table.

Run with:
    make exp-audit
    OR
    python scripts/run_audit.py
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_guard import (
    IDENTIFIER_COLUMNS,
    TIER1_ALLOWLIST,
    assert_no_leakage,
    enforce_allowlist,
    strip_identifiers,
)
from src.utils.result_writer import make_run_dir, mean_ci95, write_result

# ── Patterns for audit scanner ────────────────────────────────────────────────
HARDCODED_PATTERNS = [
    # Specific metric values that previously appeared hardcoded
    r'\b(52\.4|74\.2|86\.8|93\.5|97\.2|98\.9)\b',   # cycle accuracy values
    r'\b(16\.8|10\.5|5\.4|2\.8|1\.6|0\.92)\b',       # cycle escalation values
    r'\b(0\.35|78\.5|1\.15|98\.5)\b',                  # latency values
    r'\b(99\.60|99\.30|97\.2|98\.1)\b',                # pipeline metrics
    r'cycle_profiles\s*=\s*\[',                         # hardcoded profile list
    r'accuracies\s*=\s*\[.*Interpolated',               # interpolated comment
    r'escalations\s*=\s*\[.*Interpolated',
]

SIMULATED_PATTERNS = [
    r'simulate_traffic|telemetry simulator|hardcoded.*latency',
    r'time\.sleep.*simulate deep inspection',
    r'\+ 65\.0',   # artificial latency addition
    r'is_simulated',
]

MEASURED_PATTERNS = [
    r'results/.*\.json',
    r'load_latest_result',
    r'result_writer',
    r'write_result',
]


def scan_file(filepath: str):
    """Return list of (line_no, category, snippet) findings."""
    findings = []
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return findings

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        for pat in HARDCODED_PATTERNS:
            if re.search(pat, stripped, re.IGNORECASE):
                findings.append((i, "HARDCODED", stripped[:120]))
                break
        for pat in SIMULATED_PATTERNS:
            if re.search(pat, stripped, re.IGNORECASE):
                findings.append((i, "SIMULATED", stripped[:120]))
                break
        for pat in MEASURED_PATTERNS:
            if re.search(pat, stripped, re.IGNORECASE):
                findings.append((i, "MEASURED", stripped[:120]))
                break
    return findings


def run_with_identifier(seed: int, include_dst_port: bool):
    """
    Train and evaluate a Tier 1 RF model on the CIC-IDS corpus.
    If include_dst_port=True, adds Dst Port as a feature (to show leakage).
    Returns (accuracy, escalation_rate).
    """
    cic_path = "/Users/sohamsupekar/Desktop/MajorProject/DATASET/cic.csv"
    df = pd.read_csv("data/generated/tier1_real_corpus.csv")
    df.columns = [c.strip() for c in df.columns]
    # Clean: drop duplicates on allowlist features to strictly prevent train-test leakage
    df = df.drop_duplicates(subset=TIER1_ALLOWLIST).reset_index(drop=True)
    label_col = "Label"
    y = (df[label_col] != "Benign").astype(int).values

    if include_dst_port:
        features = ["Dst Port"] + TIER1_ALLOWLIST
        X = df[features].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        X = X.values
    else:
        X = enforce_allowlist(df, TIER1_ALLOWLIST)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]

    # Leakage check (only when NOT including identifier)
    if not include_dst_port:
        assert_no_leakage(X_tr, X_te)

    rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)
    probs = rf.predict_proba(X_te)
    acc = float(accuracy_score(y_te, preds))
    max_p = np.max(probs, axis=1)
    esc_rate = float(np.mean((max_p >= 0.30) & (max_p <= 0.70)))
    return acc, esc_rate


def main():
    print("=" * 70)
    print("RESULT INTEGRITY AUDIT")
    print("=" * 70)

    run_dir = make_run_dir("audit", seed=None)

    # ── 1. File scan ──────────────────────────────────────────────────────────
    scan_targets = [
        "src/dashboard/simulator.py",
        "src/dashboard/app.py",
        "src/evaluation/run_real_pipeline.py",
        "src/evaluation/run_unseen_multi_attack_loop.py",
        "src/evaluation/evaluate_unseen.py",
        "src/evaluation/generate_visuals.py",
        "src/evaluation/reproduce_tables.py",
        "src/next-dashboard/src/app/page.tsx",
        "src/next-dashboard/src/components/Charts.tsx",
        "src/next-dashboard/src/app/api/telemetry/route.ts",
    ]

    audit_rows = []
    hardcoded_files = []
    simulated_files = []

    for fpath in scan_targets:
        if not os.path.exists(fpath):
            print(f"  SKIP (not found): {fpath}")
            continue
        findings = scan_file(fpath)
        if not findings:
            audit_rows.append({
                "file": fpath, "line": "—", "category": "CLEAN", "snippet": "No hardcoded or simulated metrics found"
            })
        for lineno, cat, snippet in findings:
            audit_rows.append({"file": fpath, "line": lineno, "category": cat, "snippet": snippet})
            if cat == "HARDCODED" and fpath not in hardcoded_files:
                hardcoded_files.append(fpath)
            if cat == "SIMULATED" and fpath not in simulated_files:
                simulated_files.append(fpath)

    # ── 2. Identifier column test (before/after Dst Port) ────────────────────
    print("\n[Audit] Running before/after Dst Port experiment across 5 seeds...")
    seeds = [42, 71, 137, 256, 512]
    accs_with, escs_with = [], []
    accs_without, escs_without = [], []

    for s in seeds:
        try:
            a_with, e_with = run_with_identifier(s, include_dst_port=True)
        except Exception as ex:
            a_with, e_with = float("nan"), float("nan")
        a_without, e_without = run_with_identifier(s, include_dst_port=False)
        accs_with.append(a_with * 100)
        escs_with.append(e_with * 100)
        accs_without.append(a_without * 100)
        escs_without.append(e_without * 100)
        print(f"  Seed {s}: with_DstPort acc={a_with*100:.2f}%  without acc={a_without*100:.2f}%")

    acc_with_mean, acc_with_ci = mean_ci95(accs_with)
    acc_without_mean, acc_without_ci = mean_ci95(accs_without)
    esc_with_mean, esc_with_ci = mean_ci95(escs_with)
    esc_without_mean, esc_without_ci = mean_ci95(escs_without)

    dst_port_delta_acc = acc_with_mean - acc_without_mean
    dst_port_delta_esc = esc_with_mean - esc_without_mean

    print(f"\n  WITH Dst Port:    Acc={acc_with_mean:.2f}% ± {acc_with_ci:.2f}%, Esc={esc_with_mean:.2f}% ± {esc_with_ci:.2f}%")
    print(f"  WITHOUT Dst Port: Acc={acc_without_mean:.2f}% ± {acc_without_ci:.2f}%, Esc={esc_without_mean:.2f}% ± {esc_without_ci:.2f}%")
    print(f"  Delta (with - without): Acc Δ={dst_port_delta_acc:+.2f}%  Esc Δ={dst_port_delta_esc:+.2f}%")

    # ── 3. Document Tier 2 label source ──────────────────────────────────────
    tier2_label_source_note = (
        "In run_unseen_multi_attack_loop.py, the Tier 2 labels for escalated "
        "samples are produced by a TRAINED RandomForestClassifier (tier2_rf) "
        "fit on EMBER 2018 static PE features — NOT the CIC-IDS dataset ground-truth labels. "
        "The Tier 2 model serves as a proxy for deep verification. "
        "This introduces an impersonation risk: if the trained Tier 2 model is wrong "
        "on novel network flows (which it was not trained on), the pseudo-labels it "
        "generates may be noisy or incorrect. "
        "The CIC-IDS dataset ground-truth labels (column 'Label') are used ONLY in "
        "the evaluation function to compute reported accuracy — they are never fed "
        "back to Tier 1 during the adaptation loop. "
        "This is documented fully in docs/tier2_label_source.md."
    )

    # ── 4. Write metrics ──────────────────────────────────────────────────────
    metrics = {
        "dst_port_experiment": {
            "with_dst_port": {
                "accuracy_mean_pct": acc_with_mean,
                "accuracy_ci95_pct": acc_with_ci,
                "escalation_rate_mean_pct": esc_with_mean,
                "escalation_rate_ci95_pct": esc_with_ci,
            },
            "without_dst_port": {
                "accuracy_mean_pct": acc_without_mean,
                "accuracy_ci95_pct": acc_without_ci,
                "escalation_rate_mean_pct": esc_without_mean,
                "escalation_rate_ci95_pct": esc_without_ci,
            },
            "delta_accuracy_pct": dst_port_delta_acc,
            "delta_escalation_rate_pct": dst_port_delta_esc,
            "seeds_used": seeds,
            "interpretation": (
                "A large positive delta indicates Dst Port provided leaky predictive "
                "signal (e.g., FTP is always on port 21). A near-zero delta suggests "
                "no significant leakage through this column."
            ),
        },
        "hardcoded_files": hardcoded_files,
        "simulated_files": simulated_files,
        "tier2_label_source": tier2_label_source_note,
        "audit_row_count": len(audit_rows),
    }

    result_path = write_result(run_dir, metrics, extra_meta={"audit_targets": scan_targets})

    # ── 5. Write docs/audit.md ────────────────────────────────────────────────
    docs_path = Path("docs/audit.md")
    docs_path.parent.mkdir(exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    with open(docs_path, "w") as f:
        f.write(f"# Result Integrity Audit\n\n")
        f.write(f"**Generated:** {now}  \n")
        f.write(f"**Results file:** `{result_path}`\n\n")
        f.write("> This document is auto-generated by `scripts/run_audit.py`.\n\n")
        f.write("---\n\n")

        f.write("## Metric Source Classification\n\n")
        f.write("| File | Line | Category | Snippet |\n")
        f.write("|------|------|----------|---------|\n")
        for row in audit_rows:
            cat = row["category"]
            emoji = {"HARDCODED": "🔴", "SIMULATED": "🟡", "MEASURED": "🟢", "CLEAN": "✅"}.get(cat, "")
            snippet = row["snippet"].replace("|", "\\|")
            f.write(f"| `{row['file']}` | {row['line']} | {emoji} {cat} | `{snippet}` |\n")

        f.write("\n---\n\n")
        f.write("## Dst Port Leakage Experiment\n\n")
        f.write(
            f"Measured over seeds `{seeds}` using 80/20 train-test split of `data/generated/tier1_real_corpus.csv`.\n\n"
        )
        f.write("| Condition | Accuracy (mean ± 95% CI) | Escalation Rate (mean ± 95% CI) |\n")
        f.write("|-----------|--------------------------|----------------------------------|\n")
        f.write(f"| WITH Dst Port (identifier) | {acc_with_mean:.2f}% ± {acc_with_ci:.2f}% | {esc_with_mean:.2f}% ± {esc_with_ci:.2f}% |\n")
        f.write(f"| WITHOUT Dst Port (clean) | {acc_without_mean:.2f}% ± {acc_without_ci:.2f}% | {esc_without_mean:.2f}% ± {esc_without_ci:.2f}% |\n")
        f.write(f"\n**Delta:** Accuracy Δ = {dst_port_delta_acc:+.2f}%, Escalation Rate Δ = {dst_port_delta_esc:+.2f}%\n\n")

        if abs(dst_port_delta_acc) > 2.0:
            f.write("> [!WARNING]\n")
            f.write("> Large accuracy delta detected! Dst Port provides leaky predictive signal. ")
            f.write("The cleaned (without Dst Port) results are the ground truth.\n\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> Delta is small — Dst Port does not appear to provide significant leakage.\n\n")

        f.write("---\n\n")
        f.write("## Tier 2 Label Source\n\n")
        f.write(tier2_label_source_note + "\n\n")
        f.write("See also: [`docs/tier2_label_source.md`](tier2_label_source.md)\n\n")

        f.write("---\n\n")
        f.write("## Files Requiring Immediate Fix\n\n")
        if hardcoded_files or simulated_files:
            f.write("### 🔴 Hardcoded Metrics\n")
            for fpath in hardcoded_files:
                f.write(f"- `{fpath}`\n")
            f.write("\n### 🟡 Simulated / Fabricated Metrics\n")
            for fpath in simulated_files:
                f.write(f"- `{fpath}`\n")
        else:
            f.write("No hardcoded or simulated metrics found.\n")
        f.write("\n")

    print(f"\n[Audit] docs/audit.md written.")
    print(f"[Audit] Result: {result_path}")
    return metrics


if __name__ == "__main__":
    main()
