"""
src/utils/result_writer.py
─────────────────────────
Central utility for writing experiment results.

RULE: Every metric in this project MUST be written through this module.
No script may print or display a metric without first persisting it here.

Output layout:
  results/<experiment>/<timestamp>_seed<seed>/
    result.json        ← all metrics
    config.yaml        ← exact config used
    run_meta.json      ← git commit, python version, dataset paths
"""
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return False


def make_run_dir(experiment_name: str, seed: Optional[int] = None) -> Path:
    """
    Create and return a timestamped result directory.
    Format: results/<experiment>/<YYYYMMDD_HHMMSS>_seed<seed>/
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    seed_str = f"_seed{seed}" if seed is not None else ""
    run_dir = Path("results") / experiment_name / f"{ts}{seed_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_result(
    run_dir: Path,
    metrics: Dict[str, Any],
    config: Optional[Dict] = None,
    extra_meta: Optional[Dict] = None,
) -> Path:
    """
    Write metrics, config, and run metadata to the run_dir.
    Returns the path to result.json.
    """
    commit = _git_commit()
    dirty = _git_dirty()

    run_meta = {
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(run_dir),
    }
    if extra_meta:
        run_meta.update(extra_meta)

    # Write run_meta.json
    meta_path = run_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    # Write config.yaml if provided
    if config is not None:
        cfg_path = run_dir / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    # Write result.json — embed meta inline for convenience
    result = {
        "experiment_meta": run_meta,
        "metrics": metrics,
    }
    result_path = run_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[ResultWriter] Saved to {result_path}")
    return result_path


def load_latest_result(experiment_name: str) -> Optional[Dict]:
    """
    Load the most recently written result for an experiment.
    Returns None if no results exist.
    """
    base = Path("results") / experiment_name
    if not base.exists():
        return None
    runs = sorted(base.iterdir(), reverse=True)
    for run in runs:
        result_file = run / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
    return None


def load_config(config_path: str) -> Dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def mean_ci95(values):
    """Return (mean, half_width_95ci) for a list of floats."""
    import numpy as np
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if n == 1:
        return mean, float("nan")
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    # 95% CI multiplier for small n using t-distribution
    from scipy import stats
    t_val = float(stats.t.ppf(0.975, df=n - 1))
    return mean, t_val * se
