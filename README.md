# RoboCasa Latency Benchmark

This branch contains the experiment code built on top of RoboCasa for evaluating
robot policies under injected inference delay, different replan intervals, and
RTC (real-time chunk) modes.

The repository provides:

- Xiaomi-Robotics-1 and GR00T N1.5 model adapters, with hooks for pi0.5 and
  Diffusion Policy.
- Simulation-domain delay experiments whose staleness is defined in control
  steps rather than by model wall-clock latency.
- Slurm wrappers for single-GPU jobs and multi-model/multi-task sweeps.
- Incremental CSV/JSONL result writing, summary aggregation, plotting, and
  episode video re-rendering.
- Unit tests for the benchmark manifest, action conversion, and experiment
  plumbing.

This is not a replacement for the upstream RoboCasa project. It uses RoboCasa,
robosuite, MuJoCo, and model-specific checkpoints as runtime dependencies.

## Repository Layout

```text
configs/                         Benchmark manifests
scripts/xiaomi_latency_experiment.py
                                 Main evaluation runner
scripts/latency_model_adapters.py
                                 Model adapter registry
scripts/analyze_*.py             Result analysis
scripts/plot_*.py                Figures and heatmaps
scripts/replay_run_video.py      Re-render finished episodes as MP4
slurm/                           Slurm wrappers and sweep submitters
tests/                           Benchmark tests
INSTALL.md                       Detailed local environment setup
docs/stage1_xiaomi_gr00t_7task_heatmap.md
                                 Stage 1 handoff procedure
```

Model weights, local environments, kitchen assets, logs, and generated
`eval_results/` are intentionally excluded from Git. See `INSTALL.md` for the
machine-specific setup used by the original experiments.

## Quick Start

Run from the repository root with an environment that has RoboCasa and the
required model dependencies installed:

```bash
cd /path/to/robo

# Validate the benchmark manifest without running an experiment.
.conda-env/bin/python scripts/validate_latency_manifest.py

# Run one small Xiaomi evaluation locally.
.conda-env/bin/python scripts/xiaomi_latency_experiment.py \
  --model-name xiaomi \
  --task-name PickPlaceCounterToCabinet \
  --scene-set legacy5 --object-split pretrain \
  --delays-ms 0 --delay-domain sim \
  --replan-steps-list 5 --rtc off --episodes 3 \
  --output-dir eval_results/quickstart/xiaomi
```

The runner writes:

```text
episodes.csv / episodes.jsonl   Per-episode records
summary.csv / summary.json       Per-condition aggregates
```

Use `--dry-run` to inspect the Cartesian product of tasks, delays, replan
intervals, and RTC modes without loading a model.

## Slurm

The main wrapper requests one GPU, 8 CPUs, 32 GB RAM, and a 12-hour wall clock
by default. Every setting can be overridden with environment variables:

```bash
MODEL_NAME=xiaomi \
TASK_NAME=PickPlaceCounterToSink \
SCENE_SET=pretrain20 OBJECT_SPLIT=pretrain \
DELAYS_MS=0,100,300,500 DELAY_DOMAIN=sim \
REPLAN_STEPS_LIST=1,2,3,4,5,6,8,10,12,15,20 \
RTC=off CONTROL_FREQUENCY=20 EPISODES=3 \
OUTPUT_DIR=eval_results/stage1/xiaomi/PickPlaceCounterToSink \
  sbatch slurm/xiaomi_latency_experiment.sbatch
```

Run the same task with GR00T N1.5 by changing `MODEL_NAME`:

```bash
MODEL_NAME=gr00t_n1_5 \
TASK_NAME=PickPlaceCounterToSink \
SCENE_SET=pretrain20 OBJECT_SPLIT=pretrain \
DELAYS_MS=0,100,300,500 DELAY_DOMAIN=sim \
REPLAN_STEPS_LIST=1,2,3,4,5,6,8,10,12,15,20 \
RTC=off EPISODES=3 \
OUTPUT_DIR=eval_results/stage1/gr00t_n1_5/PickPlaceCounterToSink \
  sbatch slurm/xiaomi_latency_experiment.sbatch
```

Monitor and stop jobs with:

```bash
squeue -u "$USER"
scancel JOB_ID
```

For the complete seven-task Stage 1 procedure, including skip/resume rules and
validation checks, see
`docs/stage1_xiaomi_gr00t_7task_heatmap.md`.

## Plotting

Once both models have completed a task, generate a delay-by-replan heatmap:

```bash
.conda-env/bin/python scripts/plot_stage1_delay_replan_heatmap.py \
  --run Xiaomi eval_results/stage1/xiaomi/PickPlaceCounterToSink \
  --run GR00T eval_results/stage1/gr00t_n1_5/PickPlaceCounterToSink \
  --task PickPlaceCounterToSink \
  --episodes-per-cell 3 \
  --output eval_results/stage1/PickPlaceCounterToSink_xiaomi_gr00t.png
```

The plotting script writes both PNG and PDF files beside the requested output.
For the multi-model delay grid, use `scripts/analyze_sim_delay_grid.py` followed
by `scripts/plot_sim_delay_grid.py`.

## Episode Videos

The main runner can record an MP4 per episode with `--record-video` and
`--video-dir`. More commonly, re-render selected episodes from an existing run:

```bash
.conda-env/bin/python scripts/replay_run_video.py \
  --run eval_results/stage1/xiaomi/PickPlaceCounterToSink \
  --out eval_results/stage1/videos/xiaomi \
  --limit 2
```

Replay is a new rollout using the stored task, seed, condition, and scene; it is
not a bit-for-bit reconstruction of the original trajectory.

## Testing

Run the benchmark tests with the project environment:

```bash
.conda-env/bin/python -m pytest -q \
  tests/test_latency_benchmark_config.py \
  tests/test_xiaomi_latency_experiment.py
```

The current benchmark test suite covers manifest validation and action/runner
contracts. Full simulation tests require MuJoCo assets and a GPU-capable
runtime.

## Citation and Upstream Documentation

For RoboCasa installation, task definitions, kitchen assets, and general
simulation usage, consult the upstream project:

- https://github.com/robocasa/robocasa
- https://robocasa.ai

The benchmark-specific setup and model paths are documented in `INSTALL.md`.

## RoboDojo `pour_liquid_into_cup` Xiaomi smoke

The RoboDojo compatibility bridge and delay x replan smoke runner are included
in this branch. The one-click entry point supports both Slurm allocation and a
direct CUDA-visible-device run:

```bash
cd /fact_home/xunyuanliu/dev/robo
# Recommended: submit to gpu-scavenger; Slurm sets CUDA_VISIBLE_DEVICES.
bash scripts/robodojo_pour_liquid_one_click.sh

# Direct mode: use only the CUDA device exposed by this process.
CUDA_VISIBLE_DEVICES=0 ROBODOJO_RUN_MODE=direct \
  bash scripts/robodojo_pour_liquid_one_click.sh
```

The script is cold-start capable. Missing RoboDojo and XPolicyLab checkouts,
the isolated Xiaomi policy environment, PyTorch/Transformers dependencies, the
Xiaomi RoboCasa checkpoint, RoboDojo assets, Miniconda/Isaac Sim runtime, and
the Ubuntu 24.04 rootfs are downloaded or created under `.deps/`, `models/`,
and `.runtime/`. Existing resources are only reused after their required files
and imports pass validation. It does not change system drivers, Singularity
configuration, or physical GPU numbering. Slurm or the caller's
`CUDA_VISIBLE_DEVICES` controls GPU visibility; Isaac's `device_id=0` is only
the local ordinal inside that visibility mask.

The host must provide the non-downloadable execution primitives
`nvidia-smi`, `bwrap`, `singularity`, `unsquashfs`, `curl`, and Python >=3.10;
the Slurm mode additionally requires `sbatch`. The script fails early with the
missing command if the server does not provide one of them.

Before evaluation, the script verifies task metadata, GPU/Vulkan access, the
Xiaomi WebSocket server, and a headless Isaac SimulationApp. Isaac runs in a
user-local Ubuntu rootfs so the host glibc 2.34 limitation is avoided. Only
after these checks pass does it run the default 9-cell smoke:

```text
delay_ms:       0, 100, 300
replan_steps:  1, 5, 10
episodes/cell: 1
```

Override the smoke size or paths without editing the repository:

```bash
EPISODES=4 DELAYS_MS=0,100,300 REPLAN_STEPS_LIST=1,5,10 \
  MODEL_PATH=/path/to/xiaomi-robotics-1-robocasa \
  bash scripts/robodojo_pour_liquid_one_click.sh
```

Logs and metrics are written to
`eval_results/robodojo_xiaomi_delay_replan_oneclick/`. The compatibility
contract, observation versioning, and action conversion are documented in
`docs/robodojo_xiaomi.md`.
