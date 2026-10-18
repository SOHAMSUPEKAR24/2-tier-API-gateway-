# System Limitations & Non-Goals

This document explicitly defines the engineering boundaries, assumptions, and non-goals of this implementation, in accordance with the project integrity rules.

---

## 1. Execution Plane vs. Dynamic Sandboxing
- **What is Implemented:** Tier 2 operates as a **Deep Verification Model** (heavy, high-dimensional machine learning analyzer running character 3-5 gram TF-IDF vectorizers or 525-dimensional static PE feature extractors).
- **What is NOT Implemented:** Tier 2 is **NOT** a dynamic virtualization sandbox (e.g., CAPE Sandbox, Cuckoo Sandbox, or FireEye MVX). Untrusted code is never executed inside hypervisor virtual machines. Dynamic behavioral traces (API call hooking, registry modifications, process injection) are not collected.

## 2. Ingress Kernel Bypass & Hardware Acceleration
- **What is Implemented:** The gateway reverse proxy is implemented using asynchronous Python (FastAPI / Starlette) and Scikit-Learn / PyTorch inference backends. On standard hardware, it sustains ~30,000 req/sec in two-tier mode with p50 latency $< 0.05\text{ ms}$.
- **What is NOT Implemented:** Line-rate 100 Gbps packet processing using kernel bypass technologies (eBPF XDP, DPDK, PF_RING) or FPGA/SmartNIC offload is out of scope. In production line-rate deployments, Tier 1 feature extraction would be compiled to an Envoy C++ filter or eBPF kernel program.

## 3. Distributed State Synchronization
- **What is Implemented:** The Capacity-Aware PI Controller and SQLite telemetry stores operate at single-instance / local cluster scale.
- **What is NOT Implemented:** Multi-region distributed consensus (e.g., Raft-based model synchronization or globally distributed token buckets across multi-cloud regions) is not implemented. In multi-region deployments, retraining gates and model version promotions would be orchestrated via an enterprise model registry (e.g., MLflow, Kubeflow) rather than local file storage.

## 4. TLS Termination Dependency
- **What is Implemented:** Track A HTTP payload inspection assumes the API Gateway acts as the TLS termination point (standard architecture for Envoy, NGINX, Kong, and cloud ingress controllers).
- **What is NOT Implemented:** Inspection of encrypted payloads without TLS decryption (e.g., pure JA3/JA4 fingerprinting over encrypted TLS handshakes) is not used as the primary verifier, as application-layer SQLi and XSS payloads require decrypted HTTP text.

## 5. Adversarial Evaluation Scope
- **What is Implemented:** Formal feature-space adversarial perturbations, boundary-evasion attacks, escalation-flood denial of service, and feedback poisoning attacks.
- **What is NOT Implemented:** Problem-space byte-preserving polymorphic binary transformation (e.g., using custom LLVM obfuscation passes or polymorphic crypters that generate valid PE binaries while preserving malicious runtime functionality) was not evaluated against live antivirus suites.
