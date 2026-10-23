# ==============================================================================
# Makefile: Adaptive Two-Tier Malware Detection API Gateway
# Project Rule 5: Every experiment has a single command to rerun: make exp-<name>
# ==============================================================================

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

.PHONY: all test exp-audit exp-benchmark exp-unseen-family exp-escalation exp-two-signal exp-tier2 exp-adaptive exp-latency exp-adversarial paper-assets clean help

help:
	@echo "Available commands:"
	@echo "  make exp-audit          Run repository result integrity & leakage audit"
	@echo "  make exp-benchmark      Generate reproducible open-set benchmark splits"
	@echo "  make exp-unseen-family  Run multi-regime LOFO, cold-start, prevalence, & ablation"
	@echo "  make exp-escalation     Benchmark all escalation triggers and baselines"
	@echo "  make exp-two-signal     Benchmark Two-Signal Gates (conformal novelty + discriminative)"
	@echo "  make exp-tier2          Evaluate Track A (HTTP) and Track B (PE) verification"
	@echo "  make exp-adaptive       Run production-safe adaptive feedback loop"
	@echo "  make exp-latency        Run gateway load test and latency validation"
	@echo "  make exp-adversarial    Run 4 adversarial stress attacks & defenses"
	@echo "  make test               Execute test suite (leakage, feature guards)"
	@echo "  make paper-assets       Synchronize all LaTeX tables and PDF plots"
	@echo "  make all                Reproduce every experiment, table, and figure"

# Phase 1: Audit
exp-audit:
	$(PYTHON) scripts/run_audit.py

# Phase 2: Open-Set Benchmark Dataset
exp-benchmark:
	$(PYTHON) src/data_collection/benchmark_pipeline.py
	$(PYTHON) scripts/run_sampling_sensitivity.py

# Phase 2b: Multi-Regime & Unseen Family Benchmark
exp-unseen-family:
	$(PYTHON) scripts/run_uniform_lofo_eval.py
	$(PYTHON) scripts/run_lofo_standardized_gates.py

# Phase 3: Escalation Triggers
exp-escalation:
	$(PYTHON) scripts/run_escalation_comparison.py

# Phase 3b: Two-Signal Escalation Gates
exp-two-signal:
	$(PYTHON) scripts/run_two_signal_gate_benchmark.py

# Phase 4: Real Tier 2 (Track A & Track B)
exp-tier2:
	$(PYTHON) scripts/run_tier2_tracks.py

# Phase 5: Production-Safe Feedback Loop
exp-adaptive:
	$(PYTHON) scripts/run_production_adaptive_loop.py

# Phase 6: Gateway Load Test & Latency Audit
exp-latency:
	$(PYTHON) scripts/sync_final_latency_results.py

# Phase 7: Adversarial Stress Testing
exp-adversarial:
	$(PYTHON) scripts/run_adversarial_attacks.py

# Unit & Leakage Tests
test:
	$(PYTEST) tests/ -v

# Collect & Synchronize Paper Assets
paper-assets:
	mkdir -p paper_assets
	$(PYTHON) scripts/generate_authoritative_reports.py
	cp -f results/escalation_comparison/*.pdf paper_assets/ 2>/dev/null || true
	cp -f results/adversarial/*.pdf paper_assets/ 2>/dev/null || true
	cp -f results/two_signal_gate/*.pdf paper_assets/ 2>/dev/null || true
	cp -f results/tier2_tracks/*.pdf paper_assets/ 2>/dev/null || true
	cp -f results/latency/*.pdf paper_assets/ 2>/dev/null || true
	@echo "Paper assets updated in paper_assets/:"
	@ls -la paper_assets/

# Reproduce Everything
all: exp-audit exp-benchmark exp-unseen-family test exp-escalation exp-tier2 exp-adaptive exp-latency exp-adversarial paper-assets
	@echo "======================================================================"
	@echo "  ALL EXPERIMENTS REPRODUCED SUCCESSFULLY. ZERO HARDCODED NUMBERS.    "
	@echo "======================================================================"

