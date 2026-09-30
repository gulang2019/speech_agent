# Stage 1: Xiaomi / GR00T 7-Task Delay-Replan Heatmaps

This document is the handoff procedure for running the remaining seven RoboCasa
tasks with Xiaomi and GR00T N1.5. It is written for the current repository and
the existing Slurm wrappers.

## Scope

Do not run pi0.5 or Diffusion Policy in this stage.

The already completed task is `PickPlaceCounterToCabinet`. The remaining tasks
are:

```text
PickPlaceCounterToSink
PickPlaceCabinetToCounter
PickPlaceCounterToMicrowave
PickPlaceCounterToStove
OpenDrawer
TurnOffStove
CoffeeSetupMug
```

Models:

```text
xiaomi gr00t_n1_5
```

Experimental grid:

| Parameter | Value |
|---|---|
| Scene set | `pretrain20` |
| Object split | `pretrain` |
| Delay domain | `sim` |
| Injected delays | `0,100,300,500` ms |
| Replan steps | `1,2,3,4,5,6,8,10,12,15,20` |
| RTC | `off` |
| Control frequency | `20` Hz |
| Episodes per cell | `3` |

Each `(model, task)` job contains `4 x 11 x 3 = 132` episodes. The full
seven-task extension contains 14 jobs and 1,848 episodes.

## Important Output Separation

The existing `PickPlaceCounterToCabinet` results are in:

```text
eval_results/stage1_delay_replan_heatmap_selected/
```

Write the new seven-task results to a separate root so the completed result is
not overwritten:

```text
eval_results/stage1_delay_replan_heatmap_7tasks/
```

The expected output layout is:

```text
eval_results/stage1_delay_replan_heatmap_7tasks/
  xiaomi/<task>/summary.json
  xiaomi/<task>/summary.csv
  xiaomi/<task>/episodes.csv
  xiaomi/<task>/episodes.jsonl
  gr00t_n1_5/<task>/summary.json
  gr00t_n1_5/<task>/summary.csv
  gr00t_n1_5/<task>/episodes.csv
  gr00t_n1_5/<task>/episodes.jsonl
```

## Preflight Checks

Run from the repository root:

```bash
cd /fact_home/xunyuanliu/dev/robo
test -x .conda-env/bin/python
test -f models/xiaomi-robotics-1-robocasa/config.json || true
test -f configs/latency_benchmark_v1.json
which sbatch
which squeue
```

Check that no old Xiaomi/GR00T job for the same output directory is running:

```bash
squeue -u "$USER" -o '%.18i %.28j %.10T %.10M %R'
```

Do not submit pi0.5 or Diffusion Policy jobs.

## Recommended Submission: At Most Two Jobs

The Slurm wrapper allocates one GPU, 8 CPUs, 32 GB RAM, and a 12-hour wall
clock per job. To keep resource usage bounded, submit one task pair at a time:
one Xiaomi job and one GR00T job. Wait for both to finish before submitting the
next task pair.

For one task, run:

```bash
TASK="PickPlaceCounterToSink"
ROOT_OUT="eval_results/stage1_delay_replan_heatmap_7tasks"

MODEL_NAME=xiaomi \
TASK_NAME="$TASK" \
SCENE_SET=pretrain20 \
OBJECT_SPLIT=pretrain \
DELAYS_MS=0,100,300,500 \
DELAY_DOMAIN=sim \
REPLAN_STEPS_LIST=1,2,3,4,5,6,8,10,12,15,20 \
RTC=off \
CONTROL_FREQUENCY=20 \
EPISODES=3 \
OUTPUT_DIR="$ROOT_OUT/xiaomi/$TASK" \
sbatch --job-name="s1-xiaomi-${TASK}" slurm/xiaomi_latency_experiment.sbatch

MODEL_NAME=gr00t_n1_5 \
TASK_NAME="$TASK" \
SCENE_SET=pretrain20 \
OBJECT_SPLIT=pretrain \
DELAYS_MS=0,100,300,500 \
DELAY_DOMAIN=sim \
REPLAN_STEPS_LIST=1,2,3,4,5,6,8,10,12,15,20 \
RTC=off \
CONTROL_FREQUENCY=20 \
EPISODES=3 \
OUTPUT_DIR="$ROOT_OUT/gr00t_n1_5/$TASK" \
sbatch --job-name="s1-gr00t-${TASK}" slurm/xiaomi_latency_experiment.sbatch
```

The two `sbatch` commands print the job IDs. Monitor them with:

```bash
squeue -j JOB_ID_1,JOB_ID_2 -o '%.18i %.28j %.10T %.10M %R'
```

After both jobs leave `squeue`, validate the output before moving to the next
task:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("eval_results/stage1_delay_replan_heatmap_7tasks")
for model in ("xiaomi", "gr00t_n1_5"):
    task_dirs = sorted((root / model).iterdir()) if (root / model).exists() else []
    for task_dir in task_dirs:
        summary = task_dir / "summary.json"
        if not summary.exists():
            print("MISSING", summary)
            continue
        data = json.loads(summary.read_text())
        ok = data.get("num_results") == 132 and len(data.get("conditions", [])) == 44
        print("OK" if ok else "CHECK", model, task_dir.name,
              "episodes=", data.get("num_results"),
              "conditions=", len(data.get("conditions", [])))
PY
```

Expected validation for every completed task/model pair:

```text
num_results = 132
len(conditions) = 44
```

## Batch Order

Run the seven tasks in this order unless a task needs debugging:

```text
1. PickPlaceCounterToSink
2. PickPlaceCabinetToCounter
3. PickPlaceCounterToMicrowave
4. PickPlaceCounterToStove
5. OpenDrawer
6. TurnOffStove
7. CoffeeSetupMug
```

The order is operational only. It does not change the experiment.

## Skip and Resume Rules

Before submitting a `(model, task)` job, check whether its summary is complete:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("eval_results/stage1_delay_replan_heatmap_7tasks")
for model in ("xiaomi", "gr00t_n1_5"):
    for task_dir in sorted((root / model).glob("*") if (root / model).exists() else []):
        path = task_dir / "summary.json"
        if path.exists():
            data = json.loads(path.read_text())
            print(model, task_dir.name,
                  "COMPLETE" if data.get("num_results") == 132 and len(data.get("conditions", [])) == 44 else "INCOMPLETE")
PY
```

Do not resume by appending to an incomplete output directory. The experiment
writer rewrites summary files as conditions finish, but a clean rerun should
use a new directory or remove only the incomplete directory after confirming
that no Slurm job is still writing to it.

If a job fails, inspect the matching files in `logs/`:

```bash
ls -t logs/s1-*.out logs/s1-*.err logs/xiaomi-latency-*.out logs/xiaomi-latency-*.err 2>/dev/null | head
tail -n 80 logs/xiaomi-latency-JOB_ID.err
```

Rerun only the failed `(model, task)` pair. Do not rerun completed pairs.

## Per-Task Heatmaps

The plotting script accepts a model directory and task name. Once both models
for a task are complete:

```bash
TASK="PickPlaceCounterToSink"
python scripts/plot_stage1_delay_replan_heatmap.py \
  --run Xiaomi "eval_results/stage1_delay_replan_heatmap_7tasks/xiaomi/$TASK" \
  --run GR00T "eval_results/stage1_delay_replan_heatmap_7tasks/gr00t_n1_5/$TASK" \
  --task "$TASK" \
  --episodes-per-cell 3 \
  --output "eval_results/stage1_delay_replan_heatmap_7tasks/${TASK}_xiaomi_gr00t.png"
```

This also writes a PDF beside the PNG. Repeat the command for each of the seven
tasks.

## Final Completeness Check

After all jobs finish, this command must print 14 `OK` lines:

```bash
python - <<'PY'
import json
from pathlib import Path

tasks = [
    "PickPlaceCounterToSink",
    "PickPlaceCabinetToCounter",
    "PickPlaceCounterToMicrowave",
    "PickPlaceCounterToStove",
    "OpenDrawer",
    "TurnOffStove",
    "CoffeeSetupMug",
]
root = Path("eval_results/stage1_delay_replan_heatmap_7tasks")
count = 0
for model in ("xiaomi", "gr00t_n1_5"):
    for task in tasks:
        path = root / model / task / "summary.json"
        if not path.exists():
            print("MISSING", model, task)
            continue
        data = json.loads(path.read_text())
        ok = data.get("num_results") == 132 and len(data.get("conditions", [])) == 44
        print("OK" if ok else "CHECK", model, task,
              data.get("num_results"), len(data.get("conditions", [])))
        count += int(ok)
print(f"valid_pairs={count}/14")
if count != 14:
    raise SystemExit(1)
PY
```

## Interpretation Notes

- This is a first-stage screening experiment, not a final benchmark.
- Each heatmap cell has only 3 episodes, so displayed rates are restricted to
  `0%`, `33%`, `67%`, or `100%`.
- Keep all task/model pairs on the same configuration listed above.
- Do not mix these results with the older `eval_results/simdelay_grid/` data:
  that directory uses `replan=5` and 30 episodes per condition.
- The completed `PickPlaceCounterToCabinet` heatmap remains in
  `eval_results/stage1_delay_replan_heatmap_selected/` and should be included
  separately when preparing the eventual eight-task report.
