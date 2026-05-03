# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A **discrete-event simulation framework** for speech pipeline inference with continuous batching. It models a 3-stage pipeline (STT → LM → TTS) serving concurrent multi-turn user conversations, measuring latency metrics (TTFF, TBF) and SLO violation rates under varying load.

## Running Simulations

```bash
# Run the 3-model STT→LM→TTS pipeline simulation
python dataloader.py

# Run the single-model (LM only) pipeline simulation
python single_dataloader.py
```

No build step required. Dependencies: `torch`, `numpy`, `scipy`, `matplotlib`, `datasets` (HuggingFace).

## Architecture

### Core Engine: `request_utils.py`

- **`AsyncPipelineEngine`** — main orchestrator running a continuous batching event loop
  - `_continuous_batching_scheduler()`: constructs dynamic batches separating prefill vs. decode requests
  - `run()`: async event loop that moves requests through model stages
- **`BaseModel`** — wraps a single model stage (STT/LM/TTS); simulates inference time using polynomial coefficients from `mini_vllm/perf_data.json`
  - Time formula: `t = a·B·Q² + b·B·Q + c` where B=batch size, Q=max sequence length
- **`PipelineRequest`** — per-request state machine tracking prefill/decode status and per-stage TTNF timestamps
- **`Batch`** — metadata grouping requests for a single model invocation

### Entry Points

- **`dataloader.py`** — 3-model pipeline; SLOs: TTFF ≤ 5 s, TBF ≤ 0.02 s; configures `batch_sizes` and `lambda_rates` at the bottom
- **`single_dataloader.py`** — LM-only pipeline; stricter TTFF ≤ 2 s, higher lambda (200), skips first/last 2 conversation rounds
- **`visual.py`** — standalone plotting utilities; outputs go to `figures/`

### Performance Config

`mini_vllm/perf_data.json` holds polynomial coefficients per model size (e.g., `"1.3B"`), with separate `prefill` and `decode` entries.

### Output Files

| File | Contents |
|------|----------|
| `real_run_log.txt` | Per-run summary metrics |
| `real_run_detailed_log.txt` | Per-request latency details |
| `final_results.json` | Aggregated TTFF/TBF/violation rates |
| `long_decode_batches.log` | Batches with anomalously long decode times |
| `figures/critical/` | PNG plots of timeline and distributions |

## Key Concepts

- **TTFF** (Time To First Frame): latency from request arrival to first TTS output frame
- **TBF** (Time Between Frames): gap between consecutive TTS output frames
- **Prefill vs. Decode**: requests start in prefill phase; once the first token is generated they switch to decode phase with a running KV-cache
- **`lambda_rate`**: Poisson arrival rate (requests/second) — the primary load knob
- **`num_users`** / **`batch_size`**: control concurrency and scheduler batch ceiling
