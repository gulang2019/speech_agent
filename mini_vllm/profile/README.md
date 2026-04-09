# mini_vllm Profiling CLI

This folder contains a small profiling toolkit for `mini_vllm` that measures per-batch latency, power, and energy, and fits simple energy/latency models.

## Packaged Reproduction Flow

If you want the repo to collect the exact data needed for the two existing comparison plots on a new GPU, use the preset wrapper:

```bash
python3 -m mini_vllm.profile.repro \
  --model_name <your_model_name> \
  --output_dir runs/<hardware_name> \
  --compilation_mode stock_torch_compile \
  --cudagraph_mode none
```

By default this runs both packaged presets:
- `prefill_single_decode_sweep`: writes `prefill_single_decode_sweep_energy_per_token_vs_batch_tokens.png`
- `prefill_decode_only`: writes `profile_plot_power_vs_concurrency.png`

Each preset gets its own subdirectory under `--output_dir` with:
- the copied `batch_config.json` used for that run
- the measured `*.jsonl`
- regenerated `*.png` plots
- `run_metadata.json` with GPU, host, git, and resolved CLI arguments

Any extra flags accepted by `mini_vllm.profile.cli` are forwarded through the wrapper. For example:

```bash
python3 -m mini_vllm.profile.repro \
  --model_name <your_model_name> \
  --output_dir runs/<hardware_name> \
  --repeats 10 \
  --resume \
  --device_index 1
```

To rerender plots later without rerunning the batches:

```bash
python3 -m mini_vllm.profile.repro \
  --output_dir runs/<hardware_name> \
  --plot_only
```

To inspect the available packaged presets:

```bash
python3 -m mini_vllm.profile.repro --list_presets
```

## End-To-End On A New Machine

The exact `vllm` revision matters here. This repo currently pins `3rdparty/vllm` at commit `e1cd7a5faffd188cd204f7b54eea6cb35f787ee9` (`v0.14.0rc0-273-ge1cd7a5fa`), and the install script below installs that exact checkout instead of pulling an arbitrary wheel from PyPI.

If you are starting from GitHub on another machine, pull the branch that contains this flow first.

Fresh clone:

```bash
git clone --recurse-submodules git@github.com:gulang2019/speech_agent.git
cd speech_agent
git fetch origin
git switch profile-repro-packaging
git pull --ff-only origin profile-repro-packaging
git submodule update --init --recursive
```

Existing checkout:

```bash
git fetch origin
git switch profile-repro-packaging
git pull --ff-only origin profile-repro-packaging
git submodule update --init --recursive
```

One-command install + profile flow:

```bash
bash mini_vllm/profile/install_and_profile.sh -- \
  --model_name <your_model_name> \
  --output_dir runs/<hardware_name> \
  --compilation_mode stock_torch_compile \
  --cudagraph_mode none
```

What it does:
- initializes the pinned `3rdparty/vllm` submodule
- creates a local virtualenv at `.venvs/mini-vllm-profile`
- installs the submodule with `pip install -e ./3rdparty/vllm`
- installs plotting/runtime extras needed by the wrapper
- runs `python -m mini_vllm.profile.repro ...`

If `vllm` source install already succeeded on the machine and you only want to rerun profiling:

```bash
bash mini_vllm/profile/install_and_profile.sh --skip-install -- \
  --model_name <your_model_name> \
  --output_dir runs/<hardware_name> \
  --resume
```

The resulting `run_metadata.json` records:
- the top-level repo commit
- the pinned `3rdparty/vllm` git commit and `git describe`
- the installed Python package version/location for `vllm`, `torch`, `numpy`, and `matplotlib`

Current limitation:
- power measurement currently depends on `nvidia-smi`, so this flow is NVIDIA-only until `EnergyMeter` grows non-NVIDIA backends

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
- `numpy` and `matplotlib` for plot generation, including `--plot_only`.
- Python env with the normal `mini_vllm` runtime dependencies for full batch collection.
- Optional: `pynvml` for more robust power sampling.
- Optional: `pyyaml` if you want YAML batch configs.

## CLI Options

Common flags:
- `--plot_only`: load an existing profile file and generate plots/models without rerunning batches.
- `--input profile.jsonl`: input file for `--plot_only`. If omitted, `--output` is used as the input path.
- `--repeats 20`: repeat each batch to reduce noise.
- `--warmup 2`: minimum number of warmup runs before measurement.
- `--warmup_s 10.0`: minimum warmup time per batch before measurement, useful for DVFS ramp.
- `--warmup_scope global`: warm up once before the first measured batch. Use `batch` to warm up before every batch.
- `--idle_s 2.0`: measure idle power before batches.
- `--sample_interval_s 0.01`: power sampling interval.
- `--device_index 0`: GPU index.
- `--no_sync_cuda`: disable `torch.cuda.synchronize()` (not recommended for accurate timing).
- `--resume`: resume from an existing output file and skip completed batches.
- `--retry_failed`: when resuming, retry batches previously recorded as errors.
- `--continue_on_error`: record batch errors and continue to later batches.
- `--batch_name NAME`: only run matching batch spec name(s). Pass multiple times to select several cases.
- `--enforce_eager`: disable vLLM's compiled execution path and run eagerly. Useful as a workaround for shape-specific compile bugs.
- `--compilation_mode {none,stock_torch_compile,dynamo_trace_once,vllm_compile}`: override vLLM compilation mode.
- `--compilation_backend BACKEND`: override the torch.compile backend.
- `--cudagraph_mode {none,piecewise,full,full_decode_only,full_and_piecewise}`: override vLLM cudagraph mode.

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
- You can either provide an explicit `requests` list, or use `request_template` with `num_reqs` to replicate one request shape multiple times.
- If a batch is too large for the available KV cache or hits CUDA OOM, `--continue_on_error`
  will emit an `error` record and continue with later batches.
- Warmup uses both thresholds together: the selected warmup batch runs for at least `--warmup`
  iterations and at least `--warmup_s` seconds before timed repeats begin.
- With `--warmup_scope global` the warmup happens once on the first batch that is actually profiled.
- With `--warmup_scope batch` the warmup happens before every batch.

## Outputs

- `profile.jsonl` (or `.csv`): per-batch latency, average power, energy, plus idle baseline.
- `energy_latency_model.json`: linear models vs total tokens (if `--model_out` is set).
- `profile_plot_*.png`: scatter plots for latency and energy (if `--plot_prefix` is set).
- `profile_plot_power_vs_concurrency.png`: prefill/decode average power vs concurrency, with idle power marked at concurrency 0.
- `profile_plot_energy_per_request_vs_concurrency.png`: prefill/decode energy per request vs concurrency.
- `profile_plot_energy_per_token_vs_batch_tokens.png`: prefill/decode energy per token vs batch token count. For decode-only batches with `query_len=1`, batch token count is equal to concurrency.

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

## Example: Plot Only From Existing Profile

```bash
python -m mini_vllm.profile.cli \
  --plot_only \
  --input profile.jsonl \
  --plot_prefix profile_plot
```

This reads the existing `profile.jsonl` and regenerates the standard plots, including
`profile_plot_power_vs_concurrency.png` and
`profile_plot_energy_per_request_vs_concurrency.png` and
`profile_plot_energy_per_token_vs_batch_tokens.png`, without rerunning any batches.

## Example: Resume After OOM

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output profile.jsonl \
  --continue_on_error
```

If the run is interrupted, rerun the same command with `--resume`:

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output profile.jsonl \
  --continue_on_error \
  --resume
```

This appends new rows to `profile.jsonl`, skips already completed batches, and skips
previously failed batches unless you add `--retry_failed`.

## Example: Run One Named Case

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --batch_name decode_only_ctx1024_n1 \
  --output profile.jsonl \
  --compilation_mode stock_torch_compile \
  --cudagraph_mode none \
  --resume
```

When `--resume` is used, named batches are matched by `batch_name`, so a one-case run
can append into an existing output file without colliding with unrelated batch indices.

## Example: Prefill Single-Request + Decode Sweep

The packaged preset [presets/prefill_single_decode_sweep.json](/home/siyuanch/ssd/workspace/speech_agent/mini_vllm/profile/presets/prefill_single_decode_sweep.json) sweeps:
- prefill-only single requests with `1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 3192` prompt tokens
- decode-only batches with `n=1, 2, 4, 8, 16, 32, 64, 128` at `context_len=1024`

To run only that preset:

```bash
python3 -m mini_vllm.profile.repro \
  --preset prefill_single_decode_sweep \
  --model_name <your_model_name> \
  --output_dir runs/<hardware_name> \
  --compilation_mode stock_torch_compile \
  --cudagraph_mode none
```
