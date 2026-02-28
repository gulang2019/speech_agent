# mini_vllm Profiling CLI

This folder contains a small profiling toolkit for `mini_vllm` that measures per-batch latency, power, and energy, and fits simple energy/latency models.

## Quick Start

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output profile.jsonl
```

## Requirements

- CUDA GPU and drivers.
- `nvidia-smi` available on `PATH` for power and clock sampling.
- Python env with `mini_vllm` dependencies.
- Optional: `pynvml` for more robust power sampling.
- Optional: `pyyaml` if you want YAML batch configs.

## CLI Options

Common flags:
- `--repeats 10`: repeat each batch to reduce noise.
- `--warmup 2`: warmup runs before measurement.
- `--idle_s 2.0`: measure idle power before batches.
- `--sample_interval_s 0.01`: power sampling interval.
- `--device_index 0`: GPU index.
- `--no_sync_cuda`: disable `torch.cuda.synchronize()` (not recommended for accurate timing).

Frequency / power control (optional):
- `--graphics_clock MIN,MAX`: fix graphics clock in MHz.
- `--power_limit_w W`: set GPU power limit in watts.

Modeling / plots (optional):
- `--model_out energy_latency_model.json`: save linear models.
- `--plot_prefix profile_plot`: generate scatter plots.

## Batch Config Format

```json
[
  {
    "name": "prefill_small",
    "type": "prefill",
    "requests": [
      {"context_len": 0, "query_len": 128},
      {"context_len": 0, "query_len": 256}
    ]
  },
  {
    "name": "decode_only",
    "type": "decode",
    "requests": [
      {"context_len": 128, "query_len": 1},
      {"context_len": 256, "query_len": 1}
    ]
  },
  {
    "name": "mixed",
    "type": "mixed",
    "requests": [
      {"context_len": 0, "query_len": 256},
      {"context_len": 512, "query_len": 1}
    ]
  }
]
```

Notes:
- `type` should be one of `prefill`, `decode`, or `mixed`.
- `context_len` + `query_len` determines the required KV blocks.

## Outputs

- `profile.jsonl` (or `.csv`): per-batch latency, average power, energy, plus idle baseline.
- `energy_latency_model.json`: linear models vs total tokens (if `--model_out` is set).
- `profile_plot_*.png`: scatter plots for latency and energy (if `--plot_prefix` is set).

## Example: Fixed GPU Clock

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output profile.jsonl \
  --graphics_clock 1200,1200
```

## Example: Model + Plots

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output profile.jsonl \
  --model_out energy_latency_model.json \
  --plot_prefix profile_plot
```
