# Tier 2 Deep Verification Architecture Design

**Status:** Production Design  
**Date:** 2026-10-18  
**Scope:** Application-Layer Specialization across Track A (HTTP Payloads) and Track B (File Uploads).

---

## 1. Terminology & Semantic Grounding

In earlier iterations of the system, Tier 2 was referred to colloquially as a "sandbox". Per project rules and academic rigor:

> **Rule:** Only an isolated execution backend that dynamically executes untrusted code inside an instrumentation hypervisor (e.g., CAPE Sandbox, Cuckoo Sandbox) may be termed a **Sandbox**.  
> In contrast, heavy, stateless, multi-dimensional machine learning analyzers running on deep feature vectors are formally designated as **Deep Verification Models**.

In this architecture, Tier 2 acts as a **Deep Verification Model** that is computationally expensive ($10\times$ to $100\times$ the inference latency of Tier 1), but runs synchronously or asynchronously to resolve edge ambiguity.

---

## 2. Track A: Application Layer HTTP Payload Verification

### Motivation
Network-layer flow features (CICFlowMeter statistics) cannot inspect HTTP request semantics. Track A specializes the gateway to inspect raw HTTP traffic directly at the ingress boundary.

### Dataset
- **CSIC 2010 HTTP Dataset:** 61,065 genuine HTTP requests (36,000 Normal / Benign, 25,065 Malicious containing SQL Injection, Cross-Site Scripting, Parameter Tampering, and Buffer Overflows).
- Both Tier 1 and Tier 2 evaluate the **exact same HTTP request**.

### Feature Partitioning
```
                      ┌────────────────────────────────────────┐
                      │          Inbound HTTP Request          │
                      │  (Method, URL, Headers, Entity Body)   │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │        Tier 1         │                       │        Tier 2         │
     │ Fast Payload Features │                       │ Full-Text TF-IDF Char │
     │       (Table I)       │                       │     3-5 Grams (1500)  │
     └───────────┬───────────┘                       └───────────┬───────────┘
                 │ (0.35 ms)                                     │ (12-40 ms)
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │  Lightweight Edge RF  │                       │   Deep Verification   │
     │  (Band [0.30, 0.70])  │──[Uncertain/Escalate]─▶   Random Forest /     │
     └───────────────────────┘                       │   Neural Classifier   │
                                                     └───────────────────────┘
```

#### Tier 1 (Table I: 9 Fast Payload Features)
Extracted via lightweight regex in $<0.4$ ms:
1. `req_len`: Total byte length of URI, headers, and body.
2. `num_headers`: Count of structured HTTP header fields.
3. `special_chars`: Count of non-alphanumeric punctuation marks.
4. `sql_count`: Occurrences of canonical SQL keywords (`SELECT`, `UNION`, `OR 1=1`, etc.).
5. `script_count`: Count of script injection tags (`<script>`, `javascript:`, etc.).
6. `path_depth`: Count of directory delimiters (`/`).
7. `is_post`: Binary indicator for HTTP POST requests.
8. `ua_len`: Length of User-Agent string.
9. `consec_sym`: Count of consecutive special symbol sequences (`../`, `%20%20`, `""`).

#### Tier 2 (Deep Verification Representation)
Full-text character 3-gram to 5-gram TF-IDF vectorizer generating a 1,500-dimensional sparse feature space covering the entire serialized HTTP request, passed to a deep classifier trained on a separate split.

---

## 3. Track B: File Upload Static PE Malware Verification

### Motivation
When API endpoints receive multipart file uploads (e.g., `/api/v1/upload`), the gateway must prevent weaponized executables from passing to upstream microservices.

### Dataset & Provenance
- **EMBER 2018 Static PE Dataset:** Benchmark containing 1.0M Windows PE binary feature representations.
- **Static Analysis Guarantee:** The gateway executes **STATIC ANALYSIS ONLY**; samples are never executed, completely neutralizing weaponization risks in the inspection plane.
- **Data Provenance Clarification:** EMBER-2018 ships pre-extracted static feature dictionaries. In live deployments, the `LIEF` (Library to Instrument Executable Formats) engine extracts these identical features from raw files (clean system binaries from clean OS images vs. fresh malware samples from MalwareBazaar).

### Feature Partitioning
```
                      ┌────────────────────────────────────────┐
                      │          Inbound File Upload           │
                      │         (Windows PE Executable)        │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │        Tier 1         │                       │        Tier 2         │
     │ Fast PE Header Fields │                       │ Full EMBER Static PE  │
     │      (6 Features)     │                       │ Vector (525 Features) │
     └───────────┬───────────┘                       └───────────┬───────────┘
                 │ (0.28 ms)                                     │ (15-60 ms)
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │  Lightweight Edge RF  │                       │   Deep Verification   │
     │  (Band [0.30, 0.70])  │──[Uncertain/Escalate]─▶   Random Forest       │
     └───────────────────────┘                       │   (Full Structure)    │
                                                     └───────────────────────┘
```

#### Tier 1 Fast Features (6 Header/Entropy Dimensions)
Parsed directly from the COFF and Optional headers in $<0.3$ ms without unpacking:
1. `size`: Raw binary size in bytes.
2. `vsize`: Total virtual size of allocated sections in memory.
3. `has_signature`: Authenticode certificate table flag.
4. `has_tls`: Thread Local Storage (TLS) directory flag (common in unpackers).
5. `num_sections`: Count of declared PE sections.
6. `mean_section_entropy`: Average Shannon entropy across section headers.

#### Tier 2 Deep Features (Complete EMBER Vector Space)
A 525-dimensional static representation:
- All 6 Tier 1 header features.
- 256-bin Byte frequency histogram.
- 256-bin 2D Byte entropy distribution.
- String analysis metadata: count, average length, character entropy, path/URL/registry/MZ counts.
- Section min/max entropy and virtual size expansion ratios.

---

## 4. Latency and Compute Savings Formula

For both Track A and Track B, total gateway latency is bounded by the blended latency equation:
$$L_{\text{blended}} = (1 - \text{ER}) \cdot L_{\text{Tier 1}} + \text{ER} \cdot (L_{\text{Tier 1}} + L_{\text{Tier 2}})$$

Where:
- $\text{ER}$ is the empirical Escalation Rate ($\approx 1\% - 15\%$).
- $L_{\text{Tier 1}}$ is sub-millisecond wire-speed inspection ($0.25 - 0.45\text{ ms}$).
- $L_{\text{Tier 2}}$ is deep verification inference ($15 - 50\text{ ms}$).

By routing $>90\%$ of traffic solely through Tier 1, the two-tier gateway cuts compute load and latency by **$>85\%$** compared to a monolithic deep inspection gateway, while matching or exceeding its accuracy through the adaptive feedback loop.
