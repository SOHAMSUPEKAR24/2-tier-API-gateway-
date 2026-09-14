# Dataset Provenance and Feature Mapping

## Theoretical vs. Real-World Implementation

The original manuscript, *"Adaptive Two-Tier Malware Detection in API Gateways"*, utilizes a synthetic corpus to model Application-layer (L7) payloads, allowing for a controlled evaluation of the evidential-uncertainty mathematics and the escalation logic. 

This repository provides the **real-world implementation** of that architecture. Rather than relying on simulated JSON payloads, this implementation bridges the theoretical architecture to Network-layer (L3/L4) flow metrics for the edge screening tier (Tier 1) and static Portable Executable (PE) features for the deep analysis tier (Tier 2).

The following sections detail how the theoretical features described in the paper map to real-world datasets.

## Tier 1: Fast Edge Screening (CIC-IDS 2017/2018)

Tier 1 utilizes the **CIC-IDS** dataset to represent high-speed, inline edge traffic. We map the theoretical payload-based fast features to structural network flow anomalies.

| Paper Feature (Theoretical L7) | Real Implemented Feature (CIC-IDS L3/L4) |
| :--- | :--- |
| `payload_size_bytes` | `Fwd Packet Length Max` |
| `payload_entropy` | `Fwd Packet Length Mean` |
| `special_char_count` | `Bwd Packet Length Max` |
| `sql_keyword_count` | `Flow Bytes/s` |
| `script_tag_count` | `Flow Packets/s` |
| `base64_blob_present` | `Fwd IAT Mean` |
| `param_count` | `Bwd IAT Mean` |
| `content_type_mismatch` | `ACK Flag Count` |
| `request_rate_per_key` | `PSH Flag Count` |

## Tier 2: Deep Sandbox Inspection (EMBER 2018)

Tier 2 utilizes the **EMBER 2018** dataset to represent deep static analysis of payloads (e.g., uploaded files). We map the theoretical sandbox behaviors to static PE features parsed from the EMBER JSON representation.

| Paper Feature (Theoretical Sandbox) | Real Implemented Feature (EMBER Static PE) |
| :--- | :--- |
| `decoded_blob_entropy` | Mean of `byteentropy` array |
| `has_exe_magic_bytes` | `general.has_signature` |
| `sim_registry_writes` | `general.imports` (Proxy for execution capability) |
| `sim_network_beacons` | `strings.urls` or `general.urls` (C2 capability) |
| `sim_process_injections` | `general.exports` (Proxy for execution capability) |

## Sampling Methodology

To ensure realistic evaluation without triggering Out-Of-Memory (OOM) crashes (CIC-IDS is ~6.6GB, EMBER is ~10GB), we employ strict, balanced chunk sampling:

1. **CIC-IDS (Tier 1):** The dataset is streamed in chunks of 50,000 rows. We actively filter for exactly 5,000 `Benign` rows and 5,000 Attack rows (where `Label != 'Benign'`). 
2. **EMBER (Tier 2):** The `train_features_*.jsonl` files are iterated line-by-line. We extract exactly 5,000 Benign (`label == 0`) and 5,000 Malicious (`label == 1`) records. To aggressively minimize RAM consumption, only the required keys (`byteentropy`, `general`, `strings`) are parsed and loaded.

Both extracted sets are merged into a unified 10,000-row testing corpus with a perfect 50/50 class balance.
