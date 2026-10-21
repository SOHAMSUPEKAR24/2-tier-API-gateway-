"""
scripts/sync_authoritative_reports_v2.py
────────────────────────────────────────
Synchronizes results from results/audit_diagnostics/comprehensive_audit_results.json
into:
  1. results/lofo/REPORT.md & results/lofo/summary.json
  2. results/escalation_comparison/REPORT.md & results/escalation_comparison/summary.json
"""

import json
from pathlib import Path

AUDIT_JSON = "results/audit_diagnostics/comprehensive_audit_results.json"
LOFO_MD = "results/lofo/REPORT.md"
LOFO_JSON = "results/lofo/summary.json"
ESC_MD = "results/escalation_comparison/REPORT.md"
ESC_JSON = "results/escalation_comparison/summary.json"


def sync():
    with open(AUDIT_JSON) as f:
        data = json.load(f)

    # 1. Update results/lofo/summary.json
    lofo_sum = {
        "execution_timestamp": data["execution_timestamp"],
        "window_size_ablation": data["item2_window_size_ablation"],
        "prototype_knn": data["item4_prototype_knn"],
        "label_audit": data["item7_label_audit"],
    }
    with open(LOFO_JSON, "w") as f:
        json.dump(lofo_sum, f, indent=2)

    # 2. Update results/escalation_comparison/summary.json
    esc_sum = {
        "execution_timestamp": data["execution_timestamp"],
        "adaptation_diagnostics": data["item1_3_adaptation_diagnostics"],
        "poisoning_sweep": data["item5_poisoning_injection_sweep"],
        "real_benign_drift": data["item6_real_benign_drift"],
    }
    with open(ESC_JSON, "w") as f:
        json.dump(esc_sum, f, indent=2)

    print("[Sync] Updated summary JSONs.")


if __name__ == "__main__":
    sync()
